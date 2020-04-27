import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.getcwd() + "/utils/")
from data_loader import DataLoader as Dataloader
from data_writer import DataWriter as Datawriter
from labels import label2id, id2label

print("-------------------- Now training ensemble model --------------------")

gpu = torch.cuda.is_available()

train_logits = torch.FloatTensor(np.loadtxt("./datasets/logits/BERT_large_train_logits.csv", delimiter=","))
dev_logits = torch.FloatTensor(np.loadtxt("./datasets/logits/BERT_large_dev_logits.csv", delimiter=","))
test_logits = torch.FloatTensor(np.loadtxt("./datasets/logits/BERT_large_test_logits.csv", delimiter=","))

train_cols = ['span_word_length',
 'article_one_word_counter',
 'article_span_sentence_counter',
 'word_resemble_factor',
 'word_count_span_sent']

train_features = torch.FloatTensor(np.array(pd.read_csv("./datasets/processed_data/train_data/features/train_data.csv")[train_cols]))
train_y = torch.tensor(np.array(pd.read_csv("./datasets/processed_data/train_data/features/train_data.csv")["gold_label_id"]))

dev_features = torch.FloatTensor(np.array(pd.read_csv("./datasets/processed_data/dev_data/features/dev_data.csv")[train_cols]))
test_features = torch.FloatTensor(np.array(pd.read_csv("./datasets/processed_data/test_data/features/test_data.csv")[train_cols]))

skf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

LR_model = LogisticRegression(random_state = 1)
LR_model.fit(train_features, train_y)
LR_dev_proba = torch.FloatTensor(LR_model.predict_proba(dev_features))
LR_train_proba = torch.FloatTensor(LR_model.predict_proba(train_features))
LR_test_proba = torch.FloatTensor(LR_model.predict_proba(test_features))

train_data = torch.cat([train_logits, train_features, LR_train_proba],dim=1)
dev_data = torch.cat([dev_logits, dev_features, LR_dev_proba],dim=1)
test_data = torch.cat([test_logits, test_features, LR_test_proba],dim=1)

D_in, H1, H2, H3, H4, D_out = train_data.size()[1], 500, 500, 500, 500, 14

batch_size = 32

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)

kfoldpreds = []
kfoldtrue = []

for k, (train_index, dev_index) in enumerate(skf.split(train_data, train_y)):
    if k != 0:
      continue
    NN_statelist = []

    nnmodel = torch.nn.Sequential(torch.nn.Linear(D_in, H1),
                                  torch.nn.GELU(),
                                  torch.nn.Linear(H1, H2),
                                  torch.nn.GELU(),
                                  torch.nn.Linear(H2, H3),
                                  torch.nn.GELU(),
                                  torch.nn.Linear(H3, H4),
                                  torch.nn.GELU(),
                                  torch.nn.Linear(H3, D_out))
    if gpu:
        nnmodel.cuda()

    loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")
    optimizer = torch.optim.AdamW(nnmodel.parameters(), lr = 2e-5, eps = 1e-8)

    fold_train_x = train_data[train_index]
    fold_train_y = train_y[train_index]
    fold_dev_x = train_data[dev_index]
    fold_dev_y = train_y[dev_index]

    dev_f1 = 0

    bert_score = f1_score(fold_dev_y, torch.argmax(train_logits[dev_index], dim = 1), average = "micro")
    with tqdm(total=150) as pbar:
        for w in range(150):
            pbar.update(1)
            #Evaluate
            nnmodel.eval()
            if gpu:
                fold_dev_preds = torch.argmax(nnmodel(fold_dev_x.cuda()), dim = 1).cpu()
            else:
                fold_dev_preds = torch.argmax(nnmodel(fold_dev_x), dim = 1)
            currentf1 = f1_score(fold_dev_y, fold_dev_preds, average = "micro")

            dev_f1 = currentf1

            NN_statelist.append((dev_f1, nnmodel.state_dict()))

            rand_perm = torch.randperm(len(fold_train_x))
            fold_train_x = fold_train_x[rand_perm]
            fold_train_y = fold_train_y[rand_perm]

            #Train
            nnmodel.train()
            for i in range(0, len(fold_train_x), batch_size):
                if gpu:
                    logits = nnmodel(fold_train_x[i:i+batch_size].cuda())
                else:
                    logits = nnmodel(fold_train_x[i:i+batch_size])
                if gpu:
                    loss = loss_fn(logits, fold_train_y[i:i+batch_size].cuda())
                else:
                    loss = loss_fn(logits, fold_train_y[i:i+batch_size])

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
    nnmodel.load_state_dict(sorted(NN_statelist, key = lambda x: x[0], reverse = True)[0][1])
    if gpu:
        fold_dev_preds = torch.argmax(nnmodel(fold_dev_x.cuda()), dim = 1).cpu()
    else:
        fold_dev_preds = torch.argmax(nnmodel(fold_dev_x), dim = 1)
    kfoldpreds.append(fold_dev_preds)
    kfoldtrue.append(fold_dev_y)
if gpu:
    dev_preds = [id2label[x] for x in (torch.argmax(nnmodel(dev_data.cuda()), dim = 1).detach().cpu().numpy())]
else:
    dev_preds = [id2label[x] for x in (torch.argmax(nnmodel(dev_data), dim = 1).detach().numpy())]

dataloader = Dataloader()

dev_spans = dataloader.read_spans(file_name="./datasets/dev-task-TC-template.out")

dev_spans["gold_label"] = dev_preds

datawriter = Datawriter()

datawriter.pred_writer(dev_spans,"./predictions/dev_preds.txt")

if gpu:
    test_preds = [id2label[x] for x in (torch.argmax(nnmodel(test_data.cuda()), dim = 1).detach().cpu().numpy())]
else:
    test_preds = [id2label[x] for x in (torch.argmax(nnmodel(test_data), dim = 1).detach().numpy())]

dataloader = Dataloader()

test_spans = dataloader.read_spans(file_name="./datasets/test/test-task-TC-template.out")

test_spans["gold_label"] = test_preds

datawriter = Datawriter()

datawriter.pred_writer(test_spans,"./predictions/test_preds.txt")

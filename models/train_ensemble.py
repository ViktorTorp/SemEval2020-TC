import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from collections import Counter
import sys
sys.path.insert(0, "./../utils")
from data_loader import DataLoader as Dataloader
from data_writer import DataWriter as Datawriter
from labels import label2id, id2label

gpu = torch.cuda.is_available()

train_logits = torch.FloatTensor(np.loadtxt("./../datasets/Logits and embeddings/BERT_large_logits.csv", delimiter=","))

train_cols = ['span_word_length',
 'article_one_word_counter',
 'article_span_sentence_counter',
 'word_resemble_factor',
 'word_count_span_sent']

train_features = torch.FloatTensor(np.array(pd.read_csv("./../datasets/processed_data/train_data/features/train_data.csv")[train_cols]))
train_y = torch.tensor(np.array(pd.read_csv("./../datasets/processed_data/train_data/features/train_data.csv")["gold_label_id"]))

skf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

LR_model = LogisticRegression(random_state = 1)
LR_model.fit(train_features, train_y)
LR_dev_proba = torch.FloatTensor(LR_model.predict_proba(dev_features))
LR_train_proba = torch.FloatTensor(LR_model.predict_proba(train_features))
LR_test_proba = torch.FloatTensor(LR_model.predict_proba(test_features))

train_data = torch.cat([train_logits, train_features, LR_train_proba],dim=1)

D_in, H1, H2, H3, H4, D_out = train_data.size()[1], 500, 500, 500, 500, 14

batch_size = 32

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)

kfoldpreds = []
kfoldtrue = []
for k, (train_index, dev_index) in enumerate(skf.split(train_data, train_y)):
    if k != 0:
      continue
    NN_statelist = []
    print("Fold: ", k)

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
    print(bert_score)

    for w in range(150):

        #Evaluate
        nnmodel.eval()
        if gpu:
            fold_dev_preds = torch.argmax(nnmodel(fold_dev_x.cuda()), dim = 1).cpu()
        else:
            fold_dev_preds = torch.argmax(nnmodel(fold_dev_x), dim = 1)
        currentf1 = f1_score(fold_dev_y, fold_dev_preds, average = "micro")
        #if currentf1 > runningf1:
        #  break
        #runningf1 = currentf1
        if w%20 == 0:
            print(currentf1)
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

dev_spans = dataloader.read_spans(file_name="./../datasets/dev-task-TC-template.out")

dev_spans["gold_label"] = dev_preds

datawriter = Datawriter()

datawriter.pred_writer(dev_spans, "./../predictions/dev_preds.txt")

import torch
import pandas as pd
import transformers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys
import os
sys.path.insert(0, os.getcwd() + "/utils/")
from labels import label2id, id2label

batch_size = int(input("Select batch size. Larger batch sizes may result in running out of memory. 8 is recommended."))

gpu = torch.cuda.is_available()

train_spans = pd.read_csv(os.getcwd() + "/datasets/features/train_data.csv")

train_spans["gold_label"] = train_spans["gold_label"].map(lambda x: label2id[x])

tokenizer = transformers.BertTokenizer.from_pretrained("bert-large-uncased")

tokenized_inputs = tokenizer.batch_encode_plus(train_spans["spans"], pad_to_max_length = True)["input_ids"]

tokenized_attention = tokenizer.batch_encode_plus(train_spans["spans"], pad_to_max_length = True)["attention_mask"]

tokenized_inputs = torch.tensor(tokenized_inputs)
tokenized_attention = torch.tensor(tokenized_attention)

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)

def evaluate(dev_i):
    print("-------------- Now evaluating on dev set ------------------")
    dev_inputs = tokenized_inputs[dev_i]
    dev_attention = tokenized_attention[dev_i]
    dev_labels = np.array(train_spans["gold_label"])[dev_i]
    dev_data = [dev_inputs, dev_attention]

    BertModel.eval()

    dev_preds = []

    for i in range(0, len(dev_inputs), batch_size):
        print("{} %".format(round(i*100 / len(dev_inputs), 2)))

        if gpu:
            current_batch = [x[i:i+batch_size].cuda() for x in dev_data]
        else:
            current_batch = [x[i:i+batch_size] for x in dev_data]

        if gpu:
            dev_preds.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach().cpu())
        else:
            dev_preds.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach())

    dev_preds = np.array(torch.argmax(torch.cat(dev_preds), dim = 1))

    return(f1_score(dev_labels, dev_preds, average = "micro"))

for k, (train_i, dev_i) in enumerate(kfold.split(np.array(range(len(train_spans))), train_spans["gold_label"])):

    print("Currently on fold {}".format(k))
    BertModel = transformers.BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels = 14)

    if gpu:
        BertModel.cuda()

    optimizer = torch.optim.AdamW(BertModel.parameters(),lr = 2e-5,eps = 1e-8)

    train_inputs = tokenized_inputs[train_i]
    train_attention = tokenized_attention[train_i]
    train_labels = torch.tensor(train_spans["gold_label"])[train_i]

    dev_f1 = 0
    cur_best_model = None
    while True:
        cur_dev_f1 = evaluate(dev_i)
        print("Current F1 on dev set: {}".format(cur_dev_f1))
        if cur_dev_f1 < dev_f1:
            torch.save(cur_best_model, os.getcwd() +  "/state_dicts/BERT_large/BERT_large_dev_f1={}_k={}".format(dev_f1, k))
            break
        else:
            dev_f1 = cur_dev_f1
            cur_best_model = BertModel.state_dict()



        BertModel.train()

        rand_perm = torch.randperm(len(train_inputs))
        train_inputs = train_inputs[rand_perm]
        train_attention = train_attention[rand_perm]
        train_labels = train_labels[rand_perm]

        train_data = (train_inputs, train_attention, train_labels)
        print("-------------- Now training on train set ------------------")

        for batchn, i in enumerate(range(0, len(train_inputs), batch_size)):

            if gpu:
                current_batch = [x[i:i+batch_size].cuda() for x in train_data]
            else:
                current_batch = [x[i:i+batch_size] for x in train_data]

            BertModel.zero_grad()

            loss, _ = BertModel.forward(current_batch[0], current_batch[1], labels = current_batch[2])

            print("Batch {} of {}. Loss = {}".format(batchn, round((len(train_inputs) / batch_size)), loss.detach()))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(BertModel.parameters(), 1.0)

            optimizer.step()

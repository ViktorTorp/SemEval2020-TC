import torch
import pandas as pd
import transformers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os

batch_size = int(input("Select batch size. Larger batch sizes may result in running out of memory. 8 is recommended."))

gpu = torch.cuda.is_available()

tokenizer = transformers.BertTokenizer.from_pretrained("bert-large-uncased")

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)

BertModel = transformers.BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels = 14)
if gpu:
    BertModel.cuda()

if gpu:
    BertModel.load_state_dict(torch.load([os.getcwd() + "/models/state_dicts/" + file for file in os.listdir(os.getcwd() + "/models/state_dicts/")][0]))
else:
    BertModel.load_state_dict(torch.load([os.getcwd() + "/models/state_dicts/" + file for file in os.listdir(os.getcwd() + "/models/state_dicts/")][0], map_location=torch.device("cpu")))

dev_df = pd.read_csv(os.getcwd() + "/datasets/features/dev_data.csv", sep = ",")

tokenized_inputs = tokenizer.batch_encode_plus(dev_df["spans"], pad_to_max_length = True)["input_ids"]
tokenized_attention = tokenizer.batch_encode_plus(dev_df["spans"], pad_to_max_length = True)["attention_mask"]
tokenized_inputs = torch.tensor(tokenized_inputs)
tokenized_attention = torch.tensor(tokenized_attention)

logits = []
index_tracking = []

batch_size = 8

dev_logits = []

dev_data = [tokenized_inputs, tokenized_attention]
for i in range(0, len(tokenized_inputs), batch_size):
  print("{} %".format(round(i*100 / len(tokenized_inputs), 2)))

  if gpu:
      current_batch = [x[i:i+batch_size].cuda() for x in dev_data]
  else:
      current_batch = [x[i:i+batch_size] for x in dev_data]

  if gpu:
      dev_logits.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach().cpu())
  else:
      dev_logits.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach())

logits.append(torch.cat(dev_logits))

with open(os.getcwd() + "/datasets/logits/BERT_large_dev_logits.csv", "w") as outfile:
  for _, logit in enumerate(torch.cat(logits).numpy()):
      for i, n in enumerate(logit):
        outfile.write(str(n))
        if i < 13:
          outfile.write(",")
      if _ < 6128:
        outfile.write("\n")

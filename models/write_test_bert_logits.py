import torch
import pandas as pd
import transformers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import sys

print("-------------------- Now writing test logits --------------------")

batch_size = int(sys.argv[1])

gpu = torch.cuda.is_available()

model_paths = [os.getcwd() + "/models/state_dicts/" + file for file in os.listdir(os.getcwd() + "/models/state_dicts/") if file[-1] in {"0","1","2","3","4","5","6","7","8","9"}]
model_paths = sorted(model_paths, key = lambda x: x[-1])

tokenizer = transformers.BertTokenizer.from_pretrained("bert-large-uncased")

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)

BertModel = transformers.BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels = 14)
if gpu:
    BertModel.cuda()

if gpu:
    BertModel.load_state_dict(torch.load(model_paths[0]))
else:
    BertModel.load_state_dict(torch.load(model_paths[0], map_location=torch.device("cpu")))

test_df = pd.read_csv("./datasets/processed_data/test_data/features/test_data.csv", sep = ",")

tokenized_inputs = tokenizer.batch_encode_plus(test_df["spans"], pad_to_max_length = True)["input_ids"]
tokenized_attention = tokenizer.batch_encode_plus(test_df["spans"], pad_to_max_length = True)["attention_mask"]
tokenized_inputs = torch.tensor(tokenized_inputs)
tokenized_attention = torch.tensor(tokenized_attention)

logits = []
index_tracking = []

batch_size = 8

test_logits = []

test_data = [tokenized_inputs, tokenized_attention]
with tqdm(total=len(tokenized_inputs) / batch_size) as pbar:
    for i in range(0, len(tokenized_inputs), batch_size):

        if gpu:
            current_batch = [x[i:i+batch_size].cuda() for x in test_data]
        else:
            current_batch = [x[i:i+batch_size] for x in test_data]

        if gpu:
            test_logits.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach().cpu())
        else:
            test_logits.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach())
        pbar.update(1)

logits.append(torch.cat(test_logits))

with open("./datasets/logits/BERT_large_test_logits.csv", "w") as outfile:
  for _, logit in enumerate(torch.cat(logits).numpy()):
      for i, n in enumerate(logit):
        outfile.write(str(n))
        if i < 13:
          outfile.write(",")
      if _ < 6128:
        outfile.write("\n")

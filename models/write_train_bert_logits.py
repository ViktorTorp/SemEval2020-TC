import torch
import pandas as pd
import transformers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os


gpu = torch.cuda.is_available()

batch_size = int(input("Select batch size. Larger batch sizes may result in running out of memory. 8 is recommended."))

train_df = pd.read_csv(os.getcwd() + "/datasets/features/train_data.csv")

tokenizer = transformers.BertTokenizer.from_pretrained("bert-large-uncased")

tokenized_inputs = tokenizer.batch_encode_plus(train_df["spans"], pad_to_max_length = True)["input_ids"]
tokenized_attention = tokenizer.batch_encode_plus(train_df["spans"], pad_to_max_length = True)["attention_mask"]

tokenized_inputs = torch.tensor(tokenized_inputs)
tokenized_attention = torch.tensor(tokenized_attention)

labels = torch.tensor(train_df["gold_label_id"])

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)

model_paths = [os.getcwd() + "/models/state_dicts/" + file for file in os.listdir(os.getcwd() + "/models/state_dicts/")]
model_paths = sorted(model_paths, key = lambda x: x[-1])
print(model_paths)

logits = []
index_tracking = []

BertModel = transformers.BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels = 14)
if gpu:
    BertModel.cuda()

for k, (_, train_i) in enumerate(kfold.split(np.array(range(len(train_df))), labels)):
  print("-------------- Now evaluating on train set {} ------------------".format(k))
  train_inputs = tokenized_inputs[train_i]
  train_attention = tokenized_attention[train_i]
  index_tracking.append(torch.tensor(train_i))
  train_data = [train_inputs, train_attention]

  if gpu:
      BertModel.load_state_dict(torch.load(model_paths[k]))
  else:
      BertModel.load_state_dict(torch.load(model_paths[k], map_location=torch.device("cpu")))

  BertModel.eval()

  train_logits = []
  for i in range(0, len(train_inputs), batch_size):
    print("{} %".format(round(i*100 / len(train_inputs), 2)))

    current_batch = [x[i:i+batch_size] for x in train_data]

    if gpu:
        train_logits.append(BertModel.forward(current_batch[0].cuda(), current_batch[1].cuda())[0].detach().cpu())
    else:
        train_logits.append(BertModel.forward(current_batch[0], current_batch[1])[0].detach())

  logits.append(torch.cat(train_logits))



with open(os.cwd() + "/datasets/logits/BERT_large_train_logits.csv", "w") as outfile:
  for _, logit in sorted(list(zip(torch.cat(index_tracking).numpy(),torch.cat(logits).numpy())), key = lambda x: x[0]):
      for i, n in enumerate(logit):
        outfile.write(str(n))
        if i < 13:
          outfile.write(",")
      if _ < 6128:
        outfile.write("\n")

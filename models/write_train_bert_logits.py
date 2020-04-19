import torch
import pandas as pd
import transformers
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os


gpu = torch.cuda.is_available()

batch_size = int(input("Select batch size. Larger batch sizes may result in running out of memory. 8 is recommended."))

train_df = pd.read_csv("./../datasets/processed_data/train_data/features/train_data.csv")

tokenizer = transformers.BertTokenizer.from_pretrained("bert-large-uncased")

tokenized_inputs = tokenizer.batch_encode_plus(train_df["spans"], pad_to_max_length = True)["input_ids"]
tokenized_attention = tokenizer.batch_encode_plus(train_df["spans"], pad_to_max_length = True)["attention_mask"]

tokenized_inputs = torch.tensor(tokenized_inputs)
tokenized_attention = torch.tensor(tokenized_attention)

labels = torch.tensor(train_df["gold_label_id"])

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)

model_paths = os.listdir("./state_dicts/BERT_large/")
model_paths = sorted(model_paths, key = lambda x: x[-1])



logits = []
index_tracking = []

BertModel = transformers.BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels = 14)
if gpu:
    BertModel.cuda()

for k, (_, dev_i) in enumerate(kfold.split(np.array(range(len(train_df))), labels)):
  print("-------------- Now evaluating on dev set {} ------------------".format(k))
  dev_inputs = tokenized_inputs[dev_i]
  dev_attention = tokenized_attention[dev_i]
  index_tracking.append(torch.tensor(dev_i))
  dev_data = [dev_inputs, dev_attention]

  BertModel.load_state_dict(torch.load(model_paths[k]))

  BertModel.eval()

  dev_logits = []
  for i in range(0, len(dev_inputs), batch_size):
    print("{} %".format(round(i*100 / len(dev_inputs), 2)))

    current_batch = [x[i:i+batch_size] for x in dev_data]

    if gpu:
        dev_logits.append(BertModel.forward(current_batch[0].cuda(), current_batch[1].cuda())[0].detach().cpu())
    else:
        dev_logits.append(BertModel.forward(current_batch[0], current_batch[1][0].detach())

  logits.append(torch.cat(dev_logits))



 with open("./../datasets/Logits and embeddings/BERT_large-logits.csv", "w") as outfile:
  for _, logit in sorted(list(zip(torch.cat(index_tracking).numpy(),torch.cat(logits).numpy())), key = lambda x: x[0]):
      for i, n in enumerate(logit):
        outfile.write(str(n))
        if i < 13:
          outfile.write(",")
      if _ < 6128:
        outfile.write("\n")

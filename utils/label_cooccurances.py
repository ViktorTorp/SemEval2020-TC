import sys
import os
sys.path.insert(0, os.getcwd() + "/utils/")
from labels import label2id, id2label
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from matplotlib import pyplot as plt

print("-------------------- Now producing label co-ocurrance matrix plot --------------------")

df = pd.read_csv("./datasets/processed_data/train_data/features/train_data.csv")

d = defaultdict(list)
for article_id, label in df[["article_id","gold_label"]].values:
    d[article_id].append(label)

coocurrence = defaultdict(list)
for labels in d.values():
    for label in labels:
        coocurrence[label] += labels
        coocurrence[label].remove(label)

for key, val in coocurrence.items():
    coocurrence[key] = Counter(val)

c_matrix = np.zeros((14,14))

for key, counter in coocurrence.items():
    i = label2id[key]
    for label, count in counter.items():
        j = label2id[label]
        c_matrix[i,j] += count

for i in range(10):
    c_matrix = c_matrix / np.sum(c_matrix, axis=0)
    c_matrix = np.transpose(np.transpose(c_matrix) / np.sum(np.transpose(c_matrix), axis = 0))


plt.figure(figsize=(10,10))
plt.imshow(c_matrix)
plt.xticks(range(14), range(14))
plt.yticks(range(14), range(14))
for i in range(14):
    for j in range(14):
        plt.text(j, i, round(c_matrix[i, j],2), ha="center", va="center", color="w", fontsize = 16)
plt.tight_layout()
plt.savefig("./cooccurence_matrix.png")

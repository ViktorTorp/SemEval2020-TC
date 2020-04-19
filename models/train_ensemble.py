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

train_logits = torch.FloatTensor(np.loadtxt("./../datasets/Logits and embeddings/BERT_large_logits.csv", delimiter=","))

train_cols = ['span_word_length',
 'article_one_word_counter',
 'article_span_sentence_counter',
 'word_resemble_factor',
 'word_count_span_sent']

train_features = torch.FloatTensor(np.array(pd.read_csv("./../datasets/processed_data/train_data/features/train_data.csv")[train_cols]))

train_y = torch.tensor(np.array(pd.read_csv("./../datasets/processed_data/train_data/features/train_data_v1.csv")["gold_label_id"]))

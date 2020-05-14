import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import sklearn

sys.path.insert(0, os.getcwd() + "/utils/")
from labels import label2id, id2label
from data_loader import DataLoader

loader = DataLoader()


try:
    print("-------------------- Now producing label confusion matrix plot --------------------")
    dev_labels = loader.read_spans("../datasets/dev-task-TC.labels").sort_values(["article_id", "span_start", "span_end"])
    y_true = dev_labels["gold_label"].apply(lambda label: label2id[label]).values
    dev_preds = loader.read_spans("../predictions/dev_preds.txt").sort_values(["article_id", "span_start", "span_end"])
    y_pred = dev_preds["gold_label"].apply(lambda label: label2id[label]).values

    dev_conf = confusion_matrix(y_true, y_pred)
    true_pos = np.diag(dev_conf) 
    precision = true_pos / np.sum(dev_conf, axis=0)
    recall = true_pos / np.sum(dev_conf, axis=1)
    dev_conf = np.c_[dev_conf, dev_conf.sum(1), np.ones(len(dev_conf))]
    dev_conf = np.r_[dev_conf, [dev_conf.sum(0)], [np.ones(dev_conf.shape[1])]]

    dev_conf_norm = confusion_matrix(y_true, y_pred, normalize="true")
    dev_conf_norm = np.c_[dev_conf_norm, dev_conf_norm.sum(1), np.ones(len(dev_conf_norm))]
    dev_conf_norm = np.r_[dev_conf_norm, [dev_conf_norm.sum(0)], [np.ones(dev_conf_norm.shape[1])]]
    dev_conf_norm[14,:] = 1


    plt.figure(figsize=(10,10))
    plt.imshow(dev_conf_norm)
    xlabels = [i if i != 14 else "sum" for i in range(15)] + ["Recall"]
    ylabels = [i if i != 14 else "sum" for i in range(15)] + ["Precision"]

    plt.xticks(range(16), xlabels)
    plt.yticks(range(16), ylabels)
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")

    dev_conf[:len(recall),-1] = [str(round(x,2)) for x in recall]
    dev_conf[-1,:len(precision)] = [str(round(x,2)) for x in precision]


    for i in range(16):
        for j in range(16):
            if j in [14, 15] or i in [14,15]:
                c = "black"
            else:
                c = "W"
                
            if i >= 14 or j == 15:
                if (i == 15 and j == 15) or (i == 15 and j == 14) or (i == 14 and j == 15):
                    plt.text(j, i, "-\n-", ha="center", va="center", color=c, fontsize = 12)
                else:
                    plt.text(j, i, str(dev_conf[i, j])+"\n"+ "-", ha="center", va="center", color=c, fontsize = 12)

                
            else:
                plt.text(j, i, str(dev_conf[i, j])+"\n"+ str(round(dev_conf_norm[i, j]*100,1))+"%", ha="center", va="center", color=c, fontsize = 12)
            

    plt.tight_layout()
    plt.savefig("../dev_confusion_matrix.png")
except:
    print("Failed to create development confusion file")

try:
    print("-------------------- Now producing label confusion matrix plot on train --------------------")
    train_labels = loader.read_spans("../datasets/train-task2-TC.labels")
    y_true = train_labels["gold_label"].apply(lambda label: label2id[label]).values
    train_preds = np.genfromtxt("../eval/train_crossval_preds.txt",dtype=str)
    y_pred = [label2id[label] for label in train_preds]


    train_conf = confusion_matrix(y_true, y_pred)
    true_pos = np.diag(train_conf) 
    precision = true_pos / np.sum(train_conf, axis=0)
    recall = true_pos / np.sum(train_conf, axis=1)
    train_conf = np.c_[train_conf, train_conf.sum(1), np.ones(len(train_conf))]
    train_conf = np.r_[train_conf, [train_conf.sum(0)], [np.ones(train_conf.shape[1])]]

    train_conf_norm = confusion_matrix(y_true, y_pred, normalize="true")
    train_conf_norm = np.c_[train_conf_norm, train_conf_norm.sum(1), np.ones(len(train_conf_norm))]
    train_conf_norm = np.r_[train_conf_norm, [train_conf_norm.sum(0)], [np.ones(train_conf_norm.shape[1])]]
    train_conf_norm[14,:] = 1


    plt.figure(figsize=(10,10))
    plt.imshow(train_conf_norm)
    xlabels = [i if i != 14 else "sum" for i in range(15)] + ["Recall"]
    ylabels = [i if i != 14 else "sum" for i in range(15)] + ["Precision"]

    plt.xticks(range(16), xlabels)
    plt.yticks(range(16), ylabels)
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")

    train_conf[:len(recall),-1] = [str(round(x,2)) for x in recall]
    train_conf[-1,:len(precision)] = [str(round(x,2)) for x in precision]


    for i in range(16):
        for j in range(16):
            if j in [14, 15] or i in [14,15]:
                c = "black"
            else:
                c = "W"
                
            if i >= 14 or j == 15:
                if (i == 15 and j == 15) or (i == 15 and j == 14) or (i == 14 and j == 15):
                    plt.text(j, i, "-\n-", ha="center", va="center", color=c, fontsize = 12)
                else:
                    plt.text(j, i, str(train_conf[i, j])+"\n"+ "-", ha="center", va="center", color=c, fontsize = 12)

                
            else:
                plt.text(j, i, str(train_conf[i, j])+"\n"+ str(round(train_conf_norm[i, j]*100,1))+"%", ha="center", va="center", color=c, fontsize = 12)
            

    plt.tight_layout()
    plt.savefig("../train_confusion_matrix.png")
except:
    print("Failed to create training conf matrix")
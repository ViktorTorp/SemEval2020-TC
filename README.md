# Team DiSaster at SemEval-2020 Task 11
## Combining BERT and hand-crafted features for Identifying Propaganda Techniques in News Media

by 
Anders Friis Kaas, Barbara Plank and Viktor Torp Thomsen

## Introduction
This GitHub repository contains all the code to recreate our solution and results for the SemEval 2020 Task 11 technique classification subtask (https://propaganda.qcri.org/semeval2020-task11/index.html).

## Replication guide
When the training data, development data and test data is placed in the datasets folder, as specified in the 'datasets' folder's README, run the following command:

```bash
$ bash main.sh
```

You can change the batch size for the BERT-large model in the main script by changing the value of the BATCH_SIZE variable.

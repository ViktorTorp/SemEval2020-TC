# Upgrade/required libraries
pip install -r requirements.txt
python -m spacy download en


# Load data, create features and save data-analysis visuals
python ./utils/feature_eng.py

# Create co-occurance matrix
python ./utils/label_cooccurances.py


BATCH_SIZE="8"

#Fine-tune BERT
python ./models/bert_finetuning.py $BATCH_SIZE

#Write BERT train logits
FILE=./datasets/logits/BERT_large_train_logits.csv
if test -f "$FILE"; then
    echo "$FILE already exist"
else
    python ./models/write_train_bert_logits.py $BATCH_SIZE
fi

#Write BERT development logits
FILE=./datasets/logits/BERT_large_dev_logits.csv
if test -f "$FILE"; then
    echo "$FILE already exist"
else
    python ./models/write_dev_bert_logits.py $BATCH_SIZE
fi

#Write BERT test logits
FILE=./datasets/logits/BERT_large_test_logits.csv
if test -f "$FILE"; then
    echo "$FILE already exist"
else
    python ./models/write_test_bert_logits.py $BATCH_SIZE
fi

#Train ensemble and make predictions
python ./models/train_ensemble.py

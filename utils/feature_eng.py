import os
import time
import re
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.stem import PorterStemmer 

from pre_processor import PreProcessor
from data_loader import DataLoader
from data_writer import DataWriter
from pre_processor import PreProcessor
from labels import label2id, id2label
# from feature_eng import GetFeatures

nlp = spacy.load('en')

class GetFeatures:
    def __init__ (self, df, col):
        self.df = df
        self.col = col
        self.nlp = spacy.load('en')
        self.ps = PorterStemmer()
        self.bow_cols = {}
        self.preprocessor = PreProcessor()
        
    def pos_tags(self):
        self.pos = []
        with tqdm(total=len(list(self.df.iterrows()))) as pbar:
            for idx, row in self.df.iterrows():  
                nlp_sentence = self.nlp(row[self.col])
                # self.df["pos_tags"].iloc[idx] = " ".join([token.tag_ for token in nlp_sentence])
                _line = " ".join([token.tag_ for token in nlp_sentence])
                _line = re.sub(r"[-()\"_#/@;´`'\":<>{}+=~,\.\?$]", "", _line)
                self.pos.append(_line)
                pbar.update(1)
        self.pos = np.array(self.pos)
    
    def dep_tags(self):
        self.dep = []
        with tqdm(total=len(list(self.df.iterrows()))) as pbar:
            for idx, row in self.df.iterrows():  
                nlp_sentence = self.nlp(row[self.col])
                # self.df["dep_tags"].iloc[idx] = " ".join([token.dep_ for token in nlp_sentence])
                self.dep.append(" ".join([token.dep_ for token in nlp_sentence]))
                pbar.update(1)
        self.dep = np.array(self.dep)
    
    def get_lengths(self):
        if "span_sentences" in self.df.columns:
            self.df["sentence_char_length"] =  self.df.apply(lambda x: len(x["span_sentences"]), axis=1)
            self.df["sentence_word_length"] =  self.df.apply(lambda x: len(x["span_sentences"].split()), axis=1)
        if "clean_span_sentences" in self.df.columns:
            self.df["clean_sentence_word_length"] =  self.df.apply(lambda x: len(x["clean_span_sentences"].split()), axis=1)
            self.df["clean_sentence_char_length"] =  self.df.apply(lambda x: len(x["clean_span_sentences"]), axis=1)
        if "spans" in self.df.columns:
            self.df["span_word_length"] =  self.df.apply(lambda x: len(x["spans"].split()), axis=1)
            self.df["span_char_length"] =  self.df.apply(lambda x: len(x["spans"]), axis=1)
        if "clean_spans" in self.df.columns:    
            self.df["clean_span_char_length"] =  self.df.apply(lambda x: len(x["clean_spans"]), axis=1)
            self.df["clean_span_word_length"] =  self.df.apply(lambda x: len(x["clean_spans"].split()), axis=1)
        
        
    def one_word_counter(self, train_articles):
        if "clean_span_word_length" not in self.df.columns:
            self.get_lengths()
                
        articles = list(set(self.df["article_id"]))
        self.df["article_one_word_counter"] = 0
        self.df["span_stem"] = ""
        with tqdm(total=len(list(self.df.iterrows()))) as pbar:
            for article_id in articles:
                if np.min(self.df[self.df["article_id"] == article_id]["clean_span_word_length"]) > 1:
                    pbar.update(len(self.df[self.df["article_id"] == article_id]))
                else:
                    article = self.preprocessor.clean_lines(train_articles[article_id]).split()
                    article = [self.ps.stem(word) for word in article]
                    for i, row in self.df[self.df["article_id"] == article_id].iterrows():
                        if row["clean_span_word_length"] == 1:
                            self.df["span_stem"].iloc[i] = self.ps.stem(row["clean_spans"])
                            word = self.df["span_stem"].iloc[i]
                            self.df["article_one_word_counter"].iloc[i] = article.count(word)
                        pbar.update(1)
                        
                        
    def sentence_repetition_counter(self, train_articles):
        if "clean_span_word_length" not in self.df.columns:
            self.get_lengths()
            
        articles = list(set(self.df["article_id"]))
        self.df["article_span_sentence_counter"] = 0
        
        with tqdm(total=len(list(self.df.iterrows()))) as pbar:
            for article_id in articles:
                article = train_articles[article_id]
                for i, row in self.df[self.df["article_id"] == article_id].iterrows():
                    if row["clean_span_word_length"] > 1:
                        spans = row["spans"]
                        self.df["article_span_sentence_counter"].iloc[i] = article.count(spans)
                    pbar.update(1)
                    
    
    def bows(self, bow_col="pos_tags",normalized=True):
        if bow_col == "pos_tags":
            if bow_col not in self.df.columns:
                self.df["pos_tags"] = self.pos_tags(self.col)
        elif bow_col == "dep_tags":
            if bow_col not in self.df.columns:
                self.df["dep_tags"] = self.dep_tags(self.col)
        else:
            if bow_col not in self.df.columns:
                print("{} is not in the DataFrame".format(bow_col))
        
        tokens = list(set([token for token in " ".join(self.df[bow_col]).split()]))
        token_2_id = {token: i for i, token in enumerate(tokens)}
        
        tmp_df = np.zeros((len(self.df), len(tokens)))
        
        with tqdm(total=len(list(self.df.iterrows()))) as pbar:
            for i, row in self.df.iterrows():
                line_tokens = row[bow_col].split()
                for token in line_tokens:
                    tmp_df[i, token_2_id[token]] += 1
                if normalized:
                    for token_i in range(len(tokens)):
                        tmp_df[i, token_i] /= len(line_tokens)
                pbar.update(1)
        
        tmp_df = pd.DataFrame(tmp_df, columns=tokens)
        tmp_df = tmp_df.fillna(0)
        self.df[tokens] = tmp_df
        self.bow_cols[bow_col] = tokens
        
    def get_word_resemble_factor(self):
        self.df["word_resemble_factor"] = self.df["clean_spans"].apply(lambda sent: len([x for x in sent.split()])/len(set([x for x in sent.split()])))
        
        
    def _get_count(self, row):
        if row["clean_span_word_length"] > 1:
            return row["clean_span_sentences"].count(row["clean_spans"])
        else:
            return row["clean_span_sentences"].split().count(row["clean_spans"])
        
    def get_count_span_sent(self):
        self.df["word_count_span_sent"] = self.df.apply(lambda row: self._get_count(row), 1)
    
    def get_features(self,return_pos=True, return_dep=True, return_counts=True, return_oneword_counts=True, return_sentence_repetition_counts=True, return_bows=True, return_resemble_factor=True, return_span_sent_count=True, train_articles=None, bow_col="pos_tags", normalized=True):
        if return_pos:
            print("Processessing POS")
            self.pos_tags()
            self.df["pos_tags"] = self.pos
            print("     POS tags are now included")
            print()
        if return_dep:
            print("Processessing dep")
            self.dep_tags()
            self.df["dep_tags"] = self.dep
            print("     dep tags are now included")
            print()
        if return_counts:
            print("Processessing Counts")
            self.get_lengths()
            print("     Counts are now included")
            print()
        if return_oneword_counts:
            if train_articles:
                print("Processessing one word counts")
                self.one_word_counter(train_articles)
                print("     One word counts are now included")
                print()
            else:
                print("Missing train articels")
                print()
        if return_sentence_repetition_counts:
            if train_articles:
                print("Processessing repeting sentences counts")
                self.sentence_repetition_counter(train_articles)
                print("     Repeting sentences counts are now included")
                print()
            else:
                print("Missing train articels")
                print()
        if return_bows:
            if train_articles:
                print("Processessing bows on {}".format(bow_col))
                self.bows(bow_col, normalized)
                print("     Repeting sentences counts are now included")
                print()
            else:
                print("Missing train articels")
                print()
                
        if return_resemble_factor:
            print("Processessing word resemble factor")     
            self.get_word_resemble_factor()
            print("        Done processessing word resemble factor")
            print()
    
        if return_span_sent_count:
            print("Processessing span sent count")     
            self.get_count_span_sent()
            print("        Done processessing word resemble factor")
        return self.df


# Load and transform articles
loader = DataLoader()
writer = DataWriter()
preproc = PreProcessor()

print("""###################################################################
# Load data and create features                                   #
###################################################################""")

print()
print("Loading train data")
train_spans = loader.read_spans("./datasets/train-task2-TC.labels")
train_articles = loader.read_articles("./datasets/train-articles")
print("     Done loading train data")
print("Loading test data")
dev_spans = loader.read_spans("./datasets/dev-task-TC-template.out")
dev_articles = loader.read_articles("./datasets/dev-articles")
print("     Done loading test data")

test_spans = loader.read_spans("./datasets/test/test-task-TC-template.out")
test_articles = loader.read_articles("./datasets/test/test-articles")
print("     Done loading test data")

print("Cleaning data")
train_df = preproc.get_data(train_spans, train_articles)
train_df["gold_label_id"] = train_df["gold_label"].map(lambda x: label2id[x])
dev_df = preproc.get_data(dev_spans, dev_articles)
test_df = preproc.get_data(test_spans, test_articles)
print("     Done cleaning data")
print()

# create file structure for features
if not os.path.exists('./datasets/processed_data'):
    os.makedirs('./datasets/processed_data')
    os.makedirs('./datasets/processed_data/train_data')
    os.makedirs('./datasets/processed_data/dev_data')
    os.makedirs('./datasets/processed_data/test_data')
    os.makedirs('./datasets/processed_data/train_data/features')
    os.makedirs('./datasets/processed_data/dev_data/features')
    os.makedirs('./datasets/processed_data/test_data/features')

# Get train features
if os.path.isfile('./datasets/processed_data/train_data/features/train_data.csv'):
    print("Loading train features")
    train_df = pd.read_csv("./datasets/processed_data/train_data/features/train_data.csv")
    print("     Done loading train features")
else:
    print("Creating train features")
    start_time = time.time()
    gf = GetFeatures(train_df, "spans")
    train_df = gf.get_features(return_pos=True, 
                                return_dep=False, 
                                return_counts=True, 
                                return_oneword_counts=True, 
                                return_sentence_repetition_counts=True, 
                                return_bows=True, 
                                return_resemble_factor=True,
                                return_span_sent_count=True,
                                train_articles=train_articles, 
                                bow_col="pos_tags", 
                                normalized=True)
    end_time = time.time()
    print("     Done creating train features")
    print("Time to create train features:", end_time - start_time)
    writer.save_df(train_df, "./datasets/processed_data/train_data/features/train_data.csv")
print()
# Get dev features
if os.path.isfile('./datasets/processed_data/dev_data/features/dev_data.csv'):
    print("Loading dev features")
    dev_df = pd.read_csv("./datasets/processed_data/dev_data/features/dev_data.csv")
    print("     Done loading dev features")
else:
    print("Creating dev features")
    start_time = time.time()
    gf = GetFeatures(dev_df, "spans")
    dev_df = gf.get_features(return_pos=True, 
                                return_dep=False, 
                                return_counts=True, 
                                return_oneword_counts=True, 
                                return_sentence_repetition_counts=True, 
                                return_bows=True, 
                                return_resemble_factor=True,
                                return_span_sent_count=True,
                                train_articles=dev_articles, 
                                bow_col="pos_tags", 
                                normalized=True)
    end_time = time.time()
    print("     Done creating dev features")
    print("Time to create dev features:", end_time - start_time)
    writer.save_df(dev_df, "./datasets/processed_data/dev_data/features/dev_data.csv")
    print()

# Get test features
if os.path.isfile('./datasets/processed_data/test_data/features/test_data.csv'):
    print("Loading test features")
    test_df = pd.read_csv("./datasets/processed_data/test_data/features/test_data.csv")
    print("     Done loading test features")
else:
    print("Creating test features")
    start_time = time.time()
    gf = GetFeatures(test_df, "spans")
    test_df = gf.get_features(return_pos=True, 
                                return_dep=False, 
                                return_counts=True, 
                                return_oneword_counts=True, 
                                return_sentence_repetition_counts=True, 
                                return_bows=True, 
                                return_resemble_factor=True,
                                return_span_sent_count=True,
                                train_articles=test_articles, 
                                bow_col="pos_tags", 
                                normalized=True)
    end_time = time.time()
    print("     Done creating test features")
    print("Time to create test features:", end_time - start_time)
    writer.save_df(test_df, "./datasets/processed_data/test_data/features/test_data.csv")
print()

print("creating EDA visualias")
gold_labels = list(set(train_df["gold_label"]))
num_occurences = [len(train_df[train_df["gold_label"]==label]) for label in gold_labels]
sort_idx = np.argsort(num_occurences)

overview_df = pd.DataFrame({"label": np.array(gold_labels)[sort_idx], "support":np.array(num_occurences)[sort_idx]})

overview_df["% w. 1word"] = overview_df["label"].apply(lambda label: (len(train_df[(train_df["gold_label"]==label) & (train_df["span_word_length"] == 1)])/len(train_df[(train_df["gold_label"]==label)]))*100,1)
overview_df["Avg #words"] = overview_df["label"].apply(lambda label: np.mean(train_df[train_df["gold_label"]==label]["span_word_length"]),1)
# overview_df["median #words"] = overview_df["label"].apply(lambda label: np.median(train_df[train_df["gold_label"]==label]["span_word_length"]),1)
overview_df["std #words"] = overview_df["label"].apply(lambda label: np.std(train_df[train_df["gold_label"]==label]["span_word_length"]),1)

overview_df["Avg one_word_counter"] = overview_df["label"].apply(lambda label: np.mean(train_df[(train_df["gold_label"]==label) & (train_df["span_word_length"]==1)]["article_one_word_counter"]),1)
overview_df["std one_word_counter"] = overview_df["label"].apply(lambda label: np.std(train_df[(train_df["gold_label"]==label) & (train_df["span_word_length"]==1)]["article_one_word_counter"]),1)

overview_df["Avg span_sentence_counter"] = overview_df["label"].apply(lambda label: np.mean(train_df[(train_df["gold_label"]==label) & (train_df["span_word_length"]>1)]["article_span_sentence_counter"]),1)
overview_df["std span_sentence_counter"] = overview_df["label"].apply(lambda label: np.std(train_df[(train_df["gold_label"]==label) & (train_df["span_word_length"]>1)]["article_span_sentence_counter"]),1)

test = overview_df[["label", "support"]]
o_df = overview_df.set_index("label")
test["% w. 1word"] = test["label"].apply(lambda x: "{}".format(round(o_df.loc[x, "% w. 1word"], 2)))
test["Avg #words"] = test["label"].apply(lambda x: "{} ({}))".format(round(o_df.loc[x, "Avg #words"], 2), round(o_df.loc[x, "std #words"]), 2))
test["Avg one_word_counter"] = test["label"].apply(lambda x: "{} ({}))".format(round(o_df.loc[x, "Avg one_word_counter"], 2), round(o_df.loc[x, "std one_word_counter"]), 2))
test["Avg span_sentence_counter"] = test["label"].apply(lambda x: "{} ({}))".format(round(o_df.loc[x, "Avg span_sentence_counter"], 2), round(o_df.loc[x, "std span_sentence_counter"]), 2))

test["id"] = test["label"].apply(lambda x: label2id[x])
cols = ["label","id","support", "% w. 1word", "Avg #words","Avg one_word_counter", "Avg span_sentence_counter"]

with open("data_analysis.txt", "w") as fp:
    fp.write(test[cols].sort_values("support", ascending=False).to_latex(index=False).replace("   "," ").replace("  "," ").replace("  "," ").replace("  "," "))

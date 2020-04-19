import spacy
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
nlp = spacy.load('en')
from nltk.stem import PorterStemmer 
from pre_processor import PreProcessor



import spacy
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
nlp = spacy.load('en')
from nltk.stem import PorterStemmer 
from pre_processor import PreProcessor




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



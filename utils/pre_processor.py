import re
import pandas as pd

class PreProcessor:
    def __init__(self):
        pass

    def _clean_article(self, article):
        """ clean line """
        _article = article.lower()
        _article = re.sub(r"i'm", "i am", _article)
        _article = re.sub(r"he's", "he is", _article)
        _article = re.sub(r"she's", "she is", _article)
        _article = re.sub(r"that's", "that is", _article)
        _article = re.sub(r"what's", "what is", _article)
        _article = re.sub(r"where's", "where is", _article)
        _article = re.sub(r"\'ll", " will", _article)
        _article = re.sub(r"\'ve", " have", _article)
        _article = re.sub(r"\'re", " are", _article)
        _article = re.sub(r"\'d", " would", _article)
        _article = re.sub(r"won't", "will not", _article)
        _article = re.sub(r"can't", "cannot", _article)
        # _article = re.sub(r"[-()\"#/@;´`'\":<>{}+=~|.?,]", "", _article)
        # _article = _article.replace('“','')
        # _article = _article.replace('”','')
        return _article

    def clean_articles(self, articles):
        """
        Clean and preprocess articles dictionary
        Input:
        articles: dict
            {article_id: articles}
        return:
        articles: dict
            {article_id: articles}
        """
        clean_articles = {}
        for article_id, article in articles.items():
            clean_articles[article_id] = self._clean_article(article)
        return clean_articles
    
    def clean_lines(self, line):
        """ clean lines """
        _line = line.lower().strip()
        _line = re.sub(r"i'm", "i am", _line)
        _line = re.sub(r"he's", "he is", _line)
        _line = re.sub(r"she's", "she is", _line)
        _line = re.sub(r"that's", "that is", _line)
        _line = re.sub(r"what's", "what is", _line)
        _line = re.sub(r"where's", "where is", _line)
        _line = re.sub(r"\'ll", " will", _line)
        _line = re.sub(r"\'ve", " have", _line)
        _line = re.sub(r"\'re", " are", _line)
        _line = re.sub(r"\'d", " would", _line)
        _line = re.sub(r"won't", "will not", _line)
        _line = re.sub(r"can't", "cannot", _line)
        _line = re.sub("[!#$%&\()*+'´`,-./:;<=>?@[\\]^_{|}~\n]", " ", _line)
        _line = re.sub('"', "", _line)
        _line = re.sub("'", "", _line)
        return _line
    
    def get_spans(self, spans_df, articles_dict, span_col="spans", clean_col="clean_spans", article="article_id", span_s="span_start", span_e="span_end"):
        spans = []
        clean_spans = []
        for idx, row in spans_df.iterrows():
            article_id = row[article]
            span_start = row[span_s]
            span_end = row[span_e]
            span = articles_dict[article_id][span_start:span_end].lower()
            spans.append(span.strip())
            clean_spans.append(self.clean_lines(span))
        spans_df[span_col] = spans
        spans_df[clean_col] = clean_spans
        return spans_df
    
    def get_span_sentence(self, spans_df, articles_dict):
        span_sentences = []
        clean_span_sentences = []
        for idx, row in spans_df.iterrows():
            article_id = row["article_id"]
            span_start = row["span_start"]
            span_end = row["span_end"]
            article = articles_dict[article_id]
            line_start_id = article[:span_start].count("\n")
            line_end_id = article[:span_end].count("\n")
            if line_start_id != line_end_id:
                span_sentence = "\n".join(article.split("\n")[line_start_id:line_end_id + 1]).lower()
            else:
                span_sentence = article.split("\n")[line_start_id].lower()
            if row["spans"] in span_sentence:
                span_sentences.append(span_sentence)
                clean_span_sentences.append(self.clean_lines(span_sentence))
            else:
                print(idx)
        spans_df["span_sentences"] = span_sentences
        spans_df["clean_span_sentences"] = clean_span_sentences
        return spans_df
    
    def get_data(self, spans_df, articles_dict):
        spans_df = self.get_spans(spans_df, articles_dict)
        spans_df = self.get_span_sentence(spans_df, articles_dict)
        return spans_df
    
    def preb_targets(self, spans_df, col="gold_label"):
        """
        Create dummy variables for targets.
        """
        # Get dummies for labels
        train_onehot_labels = []
        for id, row in spans_df.iterrows():
            train_onehot_labels.append([0]*14)
            train_onehot_labels[-1][int(row[col])] = 1
        return train_onehot_labels
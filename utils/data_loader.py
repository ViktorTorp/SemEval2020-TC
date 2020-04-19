import os
import codecs
import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def read_spans(self, file_name): 
        """ 
        Reads spans and labels from <file_name>

        returns :
            df with cols ["article_id","span_start","span_end","gold_label"]
        """
        data = []
        with codecs.open(file_name, "r", encoding="utf8") as f:
            for idx, row in enumerate(f.readlines()):
                try:
                    article_id, gold_label, span_start, span_end = row.strip().split("\t")
                    data.append([int(article_id), int(span_start), int(span_end), gold_label])
                except:
                    print("Except at line {}".format(idx))
        df = pd.DataFrame(data, columns=["article_id","span_start","span_end","gold_label"])
        return df

    def read_articles(self, path_to_files):
        """
        Read articles from <path_to_files> with the extension <file_extenxion>
        returns : 
            dict with articles {article_id: article}
        """
        files = os.listdir(path_to_files)
        articles = {}
        excepts = []
        for file_name in files:
            try:
                article_id = os.path.basename(file_name).split(".")[0].replace("article", "")
                with codecs.open(os.path.join(path_to_files, file_name), "r", encoding="utf8") as f:
                    articles[int(article_id)] = f.read()
            except:
                print("Except with file {}".format(file_name))
                excepts.append(file_name)
        print("Read {} files with succes and {} failed".format((len(files)- len(excepts)), len(excepts)))
        return articles


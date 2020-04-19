import pandas as pd
class DataWriter:
    def __init__(self):
        pass

    def pred_writer(self, pred_df, output_path):
        """
        Writes predictions to .txt file as specified by Semeval.

        Input:
        pred_df : pd.DataFrame
            with have following cols: ["article_id","span_start","span_end","gold_label"]
        output_path : string (optional)
            the path to the predictions folder
        """
        with open(output_path , "w") as fout:
            for i in range(len(pred_df)):
                article_id = pred_df.iloc[i]["article_id"]
                span_start = pred_df.iloc[i]["span_start"]
                span_end = pred_df.iloc[i]["span_end"]
                prediction = pred_df.iloc[i]["gold_label"]
                fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
        print("Predictions written to file " + output_path )
        
        
    def save_df(self, df, output_path):
        df.to_csv(output_path)
        print("The DF is saved at {}".format(output_path))
# Team DiSaster at SemEval-2020 Task 11
## Combining BERT and hand-crafted features for Identifying Propaganda Techniques in News Media

by 
Anders Friis Kaas, Viktor Torp Thomsen and Barbara Plank

## Introduction
This GitHub repository contains all the code needed, in order to reproduce our solution and results for the SemEval 2020 Task 11 technique classification subtask (https://propaganda.qcri.org/semeval2020-task11/index.html).

## Replication guide
When the training data, development data and test data is placed in the datasets folder, as specified in the 'datasets' folder's README, run the following command:

```bash
$ bash main.sh
```

You can change the batch size for the BERT-large model in the main script by changing the value of the BATCH_SIZE variable.


## Description of features

| Feature method  | Description |
|-----------------|-------------|
| pos_tags       | Creates an array containing all spans POS tags, using spacy |
| dep_tags       | Creates an array containing all spans DEP tags, using spacy |
| get_lengths    | appends the columns following columns to the data frame: ”sentence_char_length" (the number of characters in the sentence a spans appears), ”sentence_word_length" (the number of words in the sentence a spans appears), "clean_sentence_word_length" (the number of characters in the cleaned sentence where the  spans appears), "clean_sentence_char_length" (the number of words in the cleaned sentence where the spans appears), "span_word_length" (the number of words in a span), "span_char_length" (the number of character in a span), "clean_span_char_length" (the characters in the cleaned span) and ”clean_span_word_length" (the number of words in the cleaned span)  |
| one_word_counter | If the span is only one word, then this value represents the count of how many times the Porter stem of that word appeared in the article. Otherwise it is 0. |
| sentence_repetition_counter | If the span is more than one word long, then this value represents the count of how many times that span appeared elsewhere in the article. Otherwise it is 0. |
| bows | This function create a bag of words representation for either the pos_tags or the dep_tags, i.e. this method will create one new column for each unique pos or dep tag. If normalized = True, then the bow will be normalized |
 | get_word_resemble_factor |  is the inverse uniqueness of words in a span and is calculated as $\frac{\text{number of words in span}}{\text{number of unique words in span}}$ |
| get_count_span_sent | is the number times that a span appears within the sentence it is presented in. E.g. the span “fake news” appears twice in the sentence “it is fake news about a fake news story.”| 

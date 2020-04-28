# Utils
This folder cointains all the preprossecing files ect.

* data\_loader.py
	* This file contains class for loading the data files.
* data\_writer.py
	* This file contains a class for writing our results.
* feature\_eng.py
	* This file contains a class for creating all our features (see list of features below).
* labels.py
	* This file contains two python dictionaries with the label and their corresponding ids
* pre\_processor.py
	* This file contains a class for the pre processing and data cleaning for our solution


## Description of GetFeatures methods

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

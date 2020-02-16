import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def get_vectors(all_text, args=None):
    default_args = {"strip_accents":'unicode',"analyzer":'word', "min_df":10, 
            "stop_words":'english', "ngram_range":(1, 1), "max_features":10000}
    args = args or default_args
    print(args)
    word_vectorizer = CountVectorizer(**args)
    word_vectorizer.fit(all_text)
    word_features = word_vectorizer.transform(all_text)
    char_vectorizer = CountVectorizer( strip_accents='unicode', analyzer='char', min_df=20, stop_words='english',
            ngram_range=(2, 6), max_features=50000)
    char_vectorizer.fit(all_text)
    char_features = char_vectorizer.transform(all_text)
    features = hstack([char_features, word_features])
    return features.todense()

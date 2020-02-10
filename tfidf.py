import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def get_vectors(all_text):
    word_vectorizer = TfidfVectorizer( sublinear_tf=True, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            stop_words='english', ngram_range=(1, 1), max_features=10000)
    word_vectorizer.fit(all_text)
    word_features = word_vectorizer.transform(all_text)
    char_vectorizer = TfidfVectorizer( sublinear_tf=True, strip_accents='unicode', analyzer='char', stop_words='english',
            ngram_range=(2, 6), max_features=50000)
    char_vectorizer.fit(all_text)
    char_features = char_vectorizer.transform(all_text)
    features = hstack([char_features, word_features])
    return features.todense()

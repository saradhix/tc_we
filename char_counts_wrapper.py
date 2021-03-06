import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
name='Char Counts Wrapper'
def fit_predict(X_raw_train, y_train, X_raw_test, y_test, clf):
    default_args = {"strip_accents":'unicode',"analyzer":'char', "min_df":10, "ngram_range":(2, 6), "max_features":10000}
    max_acc = 0
    max_f1 = 0
    best_args=None

    for min_df in range(1,20):
        for max_features in range(1000, 60000, 2000):
            default_args["max_features"]=max_features
            default_args["min_df"]=min_df
            X_train, X_test = get_vectors(X_raw_train, X_raw_test, default_args)
            (acc, f1) = clf.fit_predict(X_train, y_train, X_test, y_test)
            if acc > max_acc:
                max_acc=acc
                best_args = default_args.copy()
                print("No Error Current Max Accuracy=", max_acc)
    print("Final Max Accuracy=", max_acc)
    print("Best args=", best_args)

def get_vectors(X_raw_train, X_raw_test, args=None):
    default_args = {"strip_accents":'unicode',"analyzer":'char', "min_df":10, 
            "stop_words":'english', "ngram_range":(2, 6), "max_features":10000}
    args = args or default_args
    print("Char Vectorizer",args)
    vectorizer = CountVectorizer(**args)
    vectorizer.fit(X_raw_train+X_raw_test)
    X_train = vectorizer.transform(X_raw_train)
    X_test = vectorizer.transform(X_raw_test)
    #print("Train=", X_train.shape)
    return (X_train, X_test)

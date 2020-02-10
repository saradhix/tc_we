import numpy as np
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import logreg
import rf
import xgb

models = [logreg, rf, xgb]

def main():
    data_dir = sys.argv[1]
    train_file = sys.argv[2]
    num_features = int(sys.argv[3])

    #Open training file
    train_file = os.path.join(data_dir, train_file)
    X_raw_train=[]
    y_train=[]
    print("Reading from train_file", train_file)
    fp=open(train_file, 'r')
    for line in fp:
        (label, sentence) = line.strip().split('\t')
        X_raw_train.append(sentence)
        y_train.append(int(label))
    fp.close()
    print(len(X_raw_train), len(y_train))
    train = None
    test = None
    X_raw_train = X_raw_train[:train]
    y_train = y_train[:train]
    X_all = X_raw_train
    y_all = y_train

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(X_raw_train)

    ch2 = SelectKBest(chi2, k=num_features)
    X_train = ch2.fit_transform(X_train, y_train)
    feature_names =vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    for fname in feature_names:
        print(fname)
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)


    X_train =feature_transform(X_raw_train, feature_names)
    X_test =feature_transform(X_raw_test, feature_names)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print("Shape of train=", X_train.shape, "Shape of test=", X_test.shape)
    for model in models:
        model.fit_predict(X_train, y_train, X_test, y_test)



def feature_transform(X_train, feature_names):
    X_transformed = []
    for X in X_train:
        features=[0 for i in feature_names]
        features = [ X.count(f) for f in feature_names]
        X_transformed.append(features)
    return X_transformed








    
if __name__ == "__main__":
    main()

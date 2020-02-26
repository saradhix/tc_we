import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
from scipy import sparse
import gc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import time
from itertools import product
from sklearn.utils import shuffle

models=[RandomForestClassifier(), XGBClassifier(), LinearSVC(), LogisticRegression()]
model_names = ['rf_classifier', 'xgb_classifier', 'linear_svc', 'logistic_regression']

def main():
    data_dir = sys.argv[1]
    train_file = sys.argv[2]
    try:
        wid = int(sys.argv[3])
        num_workers = int(sys.argv[4])
    except:
        wid = 0
        num_workers = 0

    #Open training file
    train_file = os.path.join(data_dir, train_file)
    print("Reading from train_file", train_file)
    df = pd.read_csv(train_file, delimiter='\t', quoting=3)  
    print(df.head)
    data_list = df.values.tolist()
    X_all=[str(x[1]) for x in data_list]
    y_all=[int(x[0]) for x in data_list]
    print("#Total=",len(X_all), len(y_all))
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
    print("#Train=",len(X_train), len(y_train))
    print("#Test=",len(X_test), len(y_test))
    find_best_params(X_train, y_train, X_test, y_test, wid, num_workers)

def find_best_params(X_train, y_train, X_test, y_test, wid, num_workers):
    best_metric=0
    word_params={}
    char_params={}
    word_max_features_start = 500
    word_max_features_end = 10000
    word_max_features_step = 1000
    char_max_features_start = 5000
    char_max_features_end = 60000
    char_max_features_step = 1000
    lowercase=True
    word_ngram_range=[(1,1),(1,2),(1,3),(1,4),(1,5),(1,6)]
    char_ngram_range=[(2,3),(2,4),(2,5),(2,6),(2,7),(3,3),(3,4),(3,5),(3,6),(4,4),(4,5),(4,6),(5,6)]
    word_max_feature_list = list(range(word_max_features_start, word_max_features_end+1,word_max_features_step))
    char_max_feature_list = list(range(char_max_features_start, char_max_features_end+1, char_max_features_step))

    if num_workers == 0:
        denom = 1
    else:
        denom = num_workers

    options = {'word_max_features': word_max_feature_list,
            'char_max_features': char_max_feature_list,
            'word_ngram_range': word_ngram_range,
            'char_ngram_range': char_ngram_range}
    all_options = [dict(zip(options, v)) for v in product(*options.values())]
    all_options = shuffle(all_options, random_state=0)
    print("Wid=", wid, "Num workers=", num_workers, "Number of options=", len(all_options)/denom)
    processed = 0
    for option in all_options:
        if num_workers > 1 and processed % num_workers != wid:
            #print("Ignoring the combination", processed)
            processed +=1
            continue
        char_params={}
        word_params={}
        char_params['max_features']=option['char_max_features']
        char_params['lowercase']=True
        char_params['analyzer']='char'
        char_params['ngram_range']=option['char_ngram_range']
        word_params['max_features']=option['word_max_features']
        word_params['lowercase']=True
        word_params['analyzer']='word'
        word_params['ngram_range']=option['word_ngram_range']
        metric, name = run_one_iteration(X_train, y_train, X_test, y_test, char_params, word_params)
        if metric > best_metric:
            best_metric = metric
            best_char_params = char_params.copy()
            best_word_params = word_params.copy()
            best_method_name = name
        processed +=1
        if processed % 1 == 0:
            print( processed, "Best=", best_metric, "Meth=", best_method_name, "char", best_char_params, "word", best_word_params)

def run_one_iteration(X_train, y_train, X_test, y_test, char_params, word_params):
    local_metric = 0
    #print("Entered roi with")
    #print("Char params=", char_params)
    #print("Word params=", word_params)
    default_word_params={'analyzer':'word'}
    default_char_params={'analyzer':'char'}
    for k, v in char_params.items():
      default_char_params[k]=v
    for k, v in word_params.items():
      default_word_params[k]=v
    
    #vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word', ngram_range=(1,3),dtype=np.float32)
    vect_word = TfidfVectorizer(**default_word_params)
    vect_char = TfidfVectorizer(**default_char_params)
    vect_word.fit(X_train+X_test)
    vect_char.fit(X_train+X_test)
    X_train_word_features = vect_word.transform(X_train)
    X_train_char_features = vect_char.transform(X_train)
    X_test_word_features = vect_word.transform(X_test)
    X_test_char_features = vect_char.transform(X_test)
    X_train_word_char_features = sparse.hstack([X_train_word_features, X_train_char_features])
    X_test_word_char_features = sparse.hstack([X_test_word_features, X_test_char_features])
    gc.collect()
    for model, name in zip(models, model_names):
        model.fit(X_train_word_char_features, y_train)
        y_pred = model.predict(X_test_word_char_features)
        f1 = f1_score(y_test, y_pred, average='macro')
        if f1 > local_metric:
            local_metric = f1
            local_method_name = name
    return local_metric, local_method_name


if __name__ == "__main__":
    main()

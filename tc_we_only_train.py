import numpy as np
import sys
import os
from tqdm import tqdm
import rf
import xgb
import rbfsvm
import logreg
import knn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf8')

embeddings = {'glove840':'libglove',
        'elmo':'libelmo',
        'elmo3':'libelmov3',
        'glovetwitter':'libglovetwitter',
        'infersent_ft':'libinfersent_ft',
        'meta':'libmeta',
        'universal':'libunivencoder',
        'universalmulti':'libunivencodermultilingual',
        'concatp':'libconcatp2',
        'fasttext':'libfasttext',
        'infersent_glove':'libinfersent_gl840',
        'skipthought':'libskipthought',
        'dependency':'libdeps',
        'lexvec':'liblexvec',
        'spacy':'libspacy',
        'tfidf':'tfidf'}

#models = [logreg, neural_network10, linearsvm, rbfsvm, rf, xgb]
models = [rf, xgb, rbfsvm]
models = [ knn]
#models = [rbfsvm]
def main():
    data_dir = sys.argv[1]
    train_file = sys.argv[2]
    embedding = sys.argv[3]
    try:
        oversample = int(sys.argv[4])
    except:
        oversample = 0


    embedding_module = embeddings[embedding]

    #Open training file
    train_file = os.path.join(data_dir, train_file)
    X_raw_train=[]
    y_train=[]
    X_raw_val=[]
    y_val=[]
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
    emb = __import__(embedding_module)
    print("Generating features")
    X_all = emb.get_vectors(X_all)
    print("Shape of the dataset=", X_all.shape)
    print(type(X_all))
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    uniq_labels = sorted(list(set(y_train)))
    print("Uniq labels=", uniq_labels)
    y_train = np.array(y_train)
    if oversample!=0:
        counts = [sum(y_train==label) for label in uniq_labels]
        print("Label counts before oversampling=", counts)
        start = time.time()
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
        end = time.time()
        print("SMOTE finished in", end-start, "seconds")
        counts = [sum(y_train_res==label) for label in uniq_labels]
        print("Label counts after oversampling=", counts)
    else:
        X_train_res = X_train
        y_train_res = y_train
        
    for model in models:
        print("Running model", model.name)
        start = time.time()
        model.fit_predict(X_train_res, y_train_res, X_test, y_test)
        end = time.time()
        print(model.name, "finished in", end-start, "seconds")


if __name__ == "__main__":
    main()

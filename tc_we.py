import numpy as np
import sys
import os
from tqdm import tqdm
import rf
import xgb
import rbfsvm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf8')

embeddings = {'glove840':'libglove',
        'elmo':'libelmo',
        'glovetwitter':'libglovetwitter',
        'infersent_ft':'libinfersent_ft',
        'meta':'libmeta',
        'universal':'libunivencoder',
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
models = [rf, xgb]
#models = [rbfsvm]
def main():
    data_dir = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    embedding = sys.argv[4]
    try:
        oversample = sys.argv[5]
    except:
        oversample = 0


    embedding_module = embeddings[embedding]

    #Open training file
    train_file = os.path.join(data_dir, train_file)
    val_file = os.path.join(data_dir, val_file)
    X_raw_train=[]
    y_train=[]
    X_raw_val=[]
    y_val=[]
    print("Reading from train_file", train_file)
    fp=open(train_file, 'r')
    for line in fp:
        #print(line)
        #print(line.strip().split('\t'))
        (label, sentence) = line.strip().split('\t')
        X_raw_train.append(sentence)
        y_train.append(int(label))
    fp.close()
    print("Reading from val_file", val_file)
    fp=open(val_file, 'r')
    for line in fp:
        #print(line.strip())
        (label, sentence) = line.strip().split('\t')
        X_raw_val.append(sentence)
        y_val.append(int(label))
    fp.close()
    print(len(X_raw_train), len(y_train))
    print(len(X_raw_val), len(y_val))
    train = None
    test = None
    X_raw_train = X_raw_train[:train]
    y_train = y_train[:train]
    X_raw_val = X_raw_val[:test]
    y_val = y_val[:test]
    X_all = X_raw_train+X_raw_val
    y_all = y_train + y_val
    emb = __import__(embedding_module)
    print("Generating features")
    X_all = emb.get_vectors(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    y_train = np.array(y_train)
    if oversample!=0:
        print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
        print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

        print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
        print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
    else:
        X_train_res = X_train
        y_train_res = y_train
        
    for model in models:
        print("Running model", model.name)
        model.fit_predict(X_train_res, y_train_res, X_test, y_test)

if __name__ == "__main__":
    main()

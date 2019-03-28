import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
name="SVM RBF kernel"

def fit_predict(X_train, y_train, X_test, y_test, args={}):
    default_C = 1.0
    default_gamma = 0.7
    default_kernel = 'linear'

    C = args.get('C', default_C)

    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)


    svc = svm.SVC().fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    pickle_file = 'svmrbf_model.pickle'
    pickle.dump(svc, open(pickle_file,"wb"))
    #print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    return y_pred


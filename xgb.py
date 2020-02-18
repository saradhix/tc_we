import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import xgboost
from sklearn.preprocessing import StandardScaler
import pickle
from time import time
name = "XGBoost"
def fit_predict(X_train, y_train, X_test, y_test, args={}):
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    #print("Running ", name)
    start=time()
    clf = xgboost.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print( confusion_matrix(y_test, y_pred))
    #print( classification_report(y_test, y_pred, digits=4))
    end = time()
    accuracy= accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    print("Accuracy=", accuracy, "F1=", f1)

    #print("Took ", end-start, "seconds")
    #Dump the model
    pickle_file = 'xgb_model.pickle'
    pickle.dump(clf, open(pickle_file,"wb"))
    return accuracy, f1


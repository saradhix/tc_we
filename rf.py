import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from time import time
name="Random Forest"
def fit_predict(X_train, y_train, X_test, y_test, args={}):
    #print("Running", name)
    start = time()
    clf =  RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print( confusion_matrix(y_test, y_pred))
    #print( classification_report(y_test, y_pred, digits=4))
    print("Accuracy=", accuracy_score(y_test, y_pred), "F1=", f1_score(y_test, y_pred, average=None))
    end = time()
    #print("Took", end-start, "seconds")
    #Dump the model
    pickle_file = 'rf_model.pickle'
    pickle.dump(clf, open(pickle_file,"wb"))
    return y_pred

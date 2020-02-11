import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
from time import time
name="Random Forest"
def fit_predict(X_train, y_train, X_test, y_test, args={}):
    print("Running", name)
    default_n_estimators = 20

    n_estimators = args.get('n_estimators', default_n_estimators)
    start = time()
    clf =  RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    end = time()
    print("Took", end-start, "seconds")
    #Dump the model
    pickle_file = 'rf_model.pickle'
    pickle.dump(clf, open(pickle_file,"wb"))
    return y_pred


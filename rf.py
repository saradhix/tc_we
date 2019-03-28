import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
name="Random Forest"
def fit_predict(X_train, y_train, X_test, y_test, args={}):
    default_n_estimators = 20

    n_estimators = args.get('n_estimators', default_n_estimators)
    clf =  RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    #Dump the model
    pickle_file = 'rf_model.pickle'
    pickle.dump(clf, open(pickle_file,"wb"))
    return y_pred


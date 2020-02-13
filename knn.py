import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from time import time
name="K-Nearest Neighbours"
def fit_predict(X_train, y_train, X_test, y_test, args={}):
    print("Running", name)
    start = time()
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    end = time()
    print("Took", end-start, "seconds")
    #Dump the model
    return y_pred


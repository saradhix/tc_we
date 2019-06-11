This set of files are used to do text classification using pretrained word/sentence embeddings.

The main classifier is in tc\_we.py. You need to supply the following arguments
## Arguments

# First argument
The directory where the train and the test files are. The train and test files 
should be in the format specified by fastai. i.e tab separated. First column is 
the class label and tab and then the complete text. Samples should be separated 
by new line. Example train.csv and test.csv have been included for reference.

# Second argument
The name of the train file

# Third argument
The name of the test file (with class labels)
You may have to split your original training dataset into train and test

# Fourth argument
The name of the word/sentence embedding you intend to use. Currently it supports
the following embedding. Please see tc\_we.py to check what embeddings it supports



## ML algorithms
Currently it supports the following algorithms
XGBoost
RandomForest
LogisticRegression
NeuralNetwork
SVM with RBF kernel


## Note on implementations
The code has been organized modularly, you can add more embedding modules and 
more classification algorithms too. Check the code and feel free to improve it!

Please raise a bug if you happen to see it.

## Note
The current implementation uses SMOTE to handle class imbalance. Please install 
imblearn python module for it to work. If you don't want SMOTE, you can comment
appropriate lines

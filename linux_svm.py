# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:29:51 2018

@author: khanhle
"""
import pandas as pd
from sklearn.svm import SVC
import sys

trn_file = sys.argv[1]
tst_file = sys.argv[2]
out_file = sys.argv[3]

num_features = 100
nb_classes = 2

# load training dataset
cv_dataset = pd.read_csv(trn_file, header=None)

# load independent dataset
ind_dataset = pd.read_csv(tst_file, header=None, delimiter=' ')


def load_data_training():
    X_trn = cv_dataset.iloc[:,1:num_features+1]
    Y_trn = cv_dataset.iloc[:,0]
    
    return X_trn, Y_trn

def load_data_ind():
    X_tst = ind_dataset.iloc[:,0:num_features]
    
    return X_tst

def libsvm_model():
    clf = SVC(C=8192, gamma=8, kernel='rbf')
    return clf

X_trn, Y_trn = load_data_training()
X_tst = load_data_ind()

# Train SVM classifier
svm_model = libsvm_model()
svm_model.fit(X_trn, Y_trn)

# Make predictions
svm_preds = svm_model.predict(X_tst)

# Write output
f=open(out_file, 'w')
for x in svm_preds:
	f.write(str(x))
f.close()

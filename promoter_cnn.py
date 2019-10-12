import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from keras.constraints import maxnorm


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print(__doc__)


#define params
trn_file = sys.argv[1]
#tst_file = sys.argv[2]

num_features = 64
num_epochs = 50
nb_classes = 2
nb_kernels = 3
nb_pools = 2
# feature_list = [0,38,82,44,83,81,33,86,37,88,85]

train = pd.read_csv(trn_file, header=None)
X = train.iloc[:,1:num_features+1]
Y = train.iloc[:,0]

#i = 4600
#plt.imshow(X[i,0], interpolation='nearest')
#print('label : ', Y[i,:])
def d_cnn_model():
    model = Sequential()
    model.add(Dropout(0.1, input_shape=(num_features,1)))
    model.add(Conv1D(32, 3, activation='softsign', input_shape=(num_features,1)))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))

    # model.add(Conv1D(64, 3, activation='softsign'))
    # # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(2))

    # model.add(Conv1D(128, 3, activation='softsign'))
    # # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(32, activation='softsign'))
    # model.add(Dropout(0.3))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

# X, Y, X1, Y1, true_labels_ind = load_data_1d()
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []

# CNN
for train, test in kfold.split(X, Y):
    model = d_cnn_model()
    # oversampling
    trn_new = np.asarray(X.iloc[train])
    tst_new = np.asarray(X.iloc[test])   
    ## evaluate the model
    model.fit(trn_new.reshape(len(trn_new),num_features,1), np_utils.to_categorical(Y.iloc[train],nb_classes), epochs=num_epochs, batch_size=20, verbose=0, class_weight='auto')
    #prediction
    predictions = model.predict_classes(tst_new.reshape(len(tst_new),num_features,1))
    true_labels_cv = np.asarray(Y.iloc[test])
    print('CV: ', confusion_matrix(true_labels_cv, predictions))

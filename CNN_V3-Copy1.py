#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer as MLB

encoder = OneHotEncoder()

pd.set_option('display.max_columns', None)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxoutDense

from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.utils import to_categorical
from keras import backend as K
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from keras.layers.normalization import BatchNormalization
from keras import initializers



# In[2]:


def read_data_small():
    X_train = pd.read_csv("data_small/X_train_small.csv",skiprows = 1,header = None)
    X_test = pd.read_csv("data_small/X_test_small.csv",skiprows = 1, header = None)
    y_train = pd.read_csv("data_small/y_train_small.csv", header=None)


    return X_train, X_test, y_train
#     return np.dstack(X_train), np.dstack(X_test), np.dstack(y_train)


# In[3]:


def read_data_big():
    X_train = pd.read_csv("data_big/X_train_big.csv",skiprows = 1,header = None)
    X_test = pd.read_csv("data_big/X_test_big.csv",skiprows = 1, header = None)
    y_train = pd.read_csv("data_big/y_train_big.csv", header=None)
    return X_train,y_train,X_test


# In[4]:


X_train, X_test, y_train = read_data_small()


# In[5]:


def data_preprocessing():
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train,X_test,y_train = read_data_small()
    

        
    for i in range(2):
        X_train[i+1] = pd.get_dummies(X_train[i+1])
        X_test[i+1]= pd.get_dummies(X_test[i+1])

    for i in range(9):
        X_train[i+8] = pd.get_dummies(X_train[i+8])
        X_test[i+8]= pd.get_dummies(X_test[i+8])
    X_train.drop([3,4,5],1,inplace = True)
    X_test.drop([3,4,5],1,inplace = True)
#     for i in range(X_train.shape[1]):
#         X_train[i].fillna(X_train[i].mean())
    
    X_train.fillna(X_train.mean(axis = 1,skipna = True),inplace = True)
    y_train.fillna(y_train.mean(axis = 1,skipna = True),inplace = True)
    X_test.fillna(X_test.mean(axis = 1,skipna = True),inplace = True)

    X_train = X_train.values
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test = X_test.values
    X_test_scaled = min_max_scaler.fit_transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled)
    X_test = pd.DataFrame(X_test_scaled)
    
    return X_train, X_test, y_train
    


# In[6]:


xtn,xte,ytn = data_preprocessing()
xtn


# In[7]:


def split_data():
    X_train, X_test, Y_train = data_preprocessing()
    x_train, x_test, y_train, y_test = train_test_split(X_train,Y_train,test_size = 0.2)
      
    return x_train.to_numpy(),y_train.to_numpy(),x_test.to_numpy(),y_test.to_numpy()

#     return x_train.to_numpy(),y_train.to_numpy(),x_test.to_numpy(),y_test.to_numpy()


# In[8]:


def generate_sample_weight(lst):
    weight = []
    for item in lst:
        if item.tolist() == [1.0,0.0,0.0]:
            weight.append(1)
        elif item.tolist() == [0.0,1.0,0.0]:
            weight.append(384)
        else:
            weight.append(864)
    return weight


# In[9]:


def cohens_kappa(y_true, y_pred):
    y_true_classes = tf.argmax(y_true, 1)
    y_pred_classes = tf.argmax(y_pred, 1)
    return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, 10)[1]


# In[54]:


def evaluate_model():
    trainX, trainy, testX, testy = split_data()
    trainy = encoder.fit_transform(trainy).toarray()
    testy = encoder.fit_transform(testy).toarray()
    sample_weight1 = np.asarray(generate_sample_weight(trainy))
    sample_weight2 = np.asarray(generate_sample_weight(testy))
    trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
    testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
    verbose,epochs,batch_size = 1, 10, 16
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features),padding = 'same',kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, padding = 'same'))
    
    model.add(BatchNormalization())

    model.add(Flatten())

#     model.add(Dense(100, activation='relu'))

#     model.add(Dense(100,activation='relu'))
#     model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())


    
    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())

#     model.add(Dense(500,activation='relu'))
#     model.add(Dropout(0.5))

    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,sample_weight = sample_weight1,validation_data = (testX,testy))
    model.save('CNN_Model.h5')


# In[ ]:


evaluate_model()


# In[ ]:





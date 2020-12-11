#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import copy
import yfinance
from collections import defaultdict
from os.path import abspath,join
current_folder = abspath('');current_folder
def mse_compare_batch(X_df , Y_df, X_validation_df , Y_validation_df , ranking):
    # number of factors that is used to do regression
    n_factors = [*range(10,len(X_df.columns)//2 + 1, 1)] [::-1]
    # store mse of different method with different number of factors
    mse_pca_calibration = [] # mean squared errors of pca with calibration data
    mse_pca_validation = [] # mean squared errors of pca with validation data
    mse_lg_calibration = [] # mean squared errors of linear regression with calibration data
    mse_lg_validation = [] # mean squared errors of linear regression with validation data
    for k in n_factors:
        # index of factors
        # the top k factors combined with bottom k factors
        index = list(X_df.columns[ranking[:k]]) + \
                     list(X_df.columns[ranking[-k:]])
        # get X data
        data = X_df.loc[:,index]


        # using pca in LinearRegression
        X = data
        y = np.array(Y_df)
        pca = PCA(n_components=5) #initial model
        X_pca = pca.fit_transform(X) # pca decomposition

        model_pca = LinearRegression()# initial predicition model
        model_pca.fit(X_pca,y) # fit model

        # record mse
        mse_pca_calibration.append(mean_squared_error(model_pca.predict(X_pca),y))
        # out of sample-validation
        # get validation X data and Y data
        X_test =  np.array(X_validation_df.loc[:,index])
        X_test_pca = pca.fit_transform(X_test)
        y_test = np.array(Y_validation_df)

        # record mse(validation)
        mse_pca_validation.append(mean_squared_error(model_pca.predict(X_test_pca),y_test))

        ## compare with the performance of linear regression without PCA

        model = LinearRegression()# initial predicition model
        model.fit(X,y) # fit model
        # record mse
        mse_lg_calibration.append(mean_squared_error(model.predict(X),y))
        mse_lg_validation.append(mean_squared_error(model.predict(X_test),y_test))
        # present mse
    mse = {'mse_pca_calibration' : mse_pca_calibration, 'mse_pca_validation' : mse_pca_validation, \
                       'mse_lg_calibration' : mse_lg_calibration, 'mse_lg_validation' : mse_lg_validation}
    mse_df = pd.DataFrame(mse , index = n_factors)
    return mse_df
def inital_model(inputs,optimizer = 'adam',loss='mean_squared_error'):
    
    ######### connecting all layers ????
    # input layer
    input_img = Input(shape=(inputs, ))
    # encoding layer ( we need to figure out how many layers we need and how many nodes for each layer)
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_img)#1
    # do we need all these layer?
    encoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)#3
    encoded = Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)#3
    decoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)#4
    decoded = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(decoded)#5
    decoded = Dense(inputs, activation= 'linear', kernel_regularizer=regularizers.l2(0.01))(decoded) #6

    ######### construct and compile 
    autoencoder = Model(input_img, decoded)
    # we need to figure out what is the best loss function and optimizer function.
    # so far adam is the most popular one on the market
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder

def get_data(X,Y, P_calibrate ,batch = 10,fraq = 0.8):

    ori_data = pd.concat([X,Y] ,axis = 1)
    data = ori_data
    # number for calibrate and validation
    N_calibrate = int( P_calibrate * X.shape[0])
    batch_N_calibrate = int(N_calibrate * fraq)
    # store and classify data using defaultdict
    X_data = defaultdict(defaultdict)
    Y_data = defaultdict(defaultdict)
    X_data['calibrate']['processed']  = data.iloc[:N_calibrate,:-1]
    Y_data['calibrate']['processed']  = data.iloc[:N_calibrate,-1]
    X_data['validation']['processed']  = data.iloc[N_calibrate:,:-1]
    Y_data['validation']['processed']  = data.iloc[N_calibrate:,-1]
    # generate data of returns direction [-1,1]
    # -1 indicate negative returns, 1 indicate positive returns
    Y_data['calibrate']['direction'] = [-1 if i < 0 else 1 for i in Y_data['calibrate']['processed'] ]
    Y_data['validation']['direction'] = [-1 if i < 0 else 1 for i in Y_data['validation']['processed'] ]

    for key in range(batch): # batch shuffle
        temp_data = ori_data.sample(frac = fraq)
        # calibrate
        X_data['calibrate']['processed_batch{}'.format(key)] = temp_data.iloc[:batch_N_calibrate,:-1]
        Y_data['calibrate']['processed_batch{}'.format(key)]  = temp_data.iloc[:batch_N_calibrate,-1]
        # validation
        X_data['validation']['processed_batch{}'.format(key)]  = temp_data.iloc[batch_N_calibrate:,:-1]
        Y_data['validation']['processed_batch{}'.format(key)]  = temp_data.iloc[batch_N_calibrate:,-1]
        # generate data of returns direction [-1,1]
        # -1 indicate negative returns, 1 indicate positive returns
        Y_data['calibrate']['direction_batch{}'.format(key)] = [-1 if i < 0 else 1 for i in Y_data['calibrate']['processed_batch{}'.format(key)] ]
        Y_data['validation']['direction_batch{}'.format(key)] = [-1 if i < 0 else 1 for i in Y_data['validation']['processed_batch{}'.format(key)] ]
    
    return X_data,Y_data

# In[ ]:


def inital_model_portfolio(inputs,optimizer = 'sgd', loss = 'mean_squared_error'):    
    input_img = Input(shape=(inputs,))
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_img)
    encoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)
    decoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)
    decoded = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(decoded)
    decoded = Dense(1, activation= 'linear', kernel_regularizer=regularizers.l2(0.01))(decoded) 
    
    
    # construct and compile
    deep_learner = Model(input_img, decoded)
    deep_learner.compile(optimizer=optimizer, loss=loss)
    return deep_learner


def process_nan(data,method = 'dropna' ):
    if method == 'dropna':
        # dropna and set batch size to 2000
        data = data.dropna()  ;frac = 2000/data.shape[0]
    elif method == 'fillna_0_after_normalization':
        # normalization then fill na with 0 and set batch size to 3000
        data = pd.concat([(data.iloc[:,:-1] - data.iloc[:,:-1].min())/(data.iloc[:,:-1].max() - data.iloc[:,:-1].min()),data.iloc[:,-1]],axis = 1) ; data = data.fillna(0) ;frac = 3000/data.shape[0]
    elif method == 'fillna_0' :
        # fill na with 0 and set batch size to 3000
        data = data.fillna(0) ;frac = 3000/data.shape[0]
    elif method == 'fillna_mean' :
        # fill na with mean and set batch size to 3000
        data = data.fillna(data.mean() ,axis = 0) ; frac = 3000/data.shape[0]
    print(data.shape)
    return data,frac

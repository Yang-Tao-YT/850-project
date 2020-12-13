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
from sklearn.preprocessing import StandardScaler , normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import copy
import yfinance
from collections import defaultdict
from os.path import abspath,join
current_folder = abspath('');current_folder
def mse_compare(X_df , Y_df, X_validation_df , Y_validation_df , ranking):

def mse_compare(X_df , Y_df, X_validation_df , Y_validation_df , ranking):
    '''
    calculate and compare the mse of linear regression and pca prediction
    Also, compare the mse of both under the different number of X factors, reduced by autoencoder
    input:
        X_df ,
        Y_df,
        X_validation_df ,
        Y_validation_df ,
        ranking from autoencoder
    output: mse_df,r_squre_df , trained model_pca, trained model_lg
    '''
    # number of factors that is used to do regression
    n_factors = [len(X_df.columns)] + [*range(10,len(X_df.columns)//2 + 1, 1)] [::-1]
    # store mse of different method with different number of factors
    mse_pca_calibration = [] # mean squared errors of pca with calibration data
    mse_pca_validation = [] # mean squared errors of pca with validation data
    mse_lg_calibration = [] # mean squared errors of linear regression with calibration data
    mse_lg_validation = [] # mean squared errors of linear regression with validation data
    r_squre_pca_calibration = [] # r squared of pca with calibration data
    r_squre_pca_validation = [] # r squared of pca with validation data
    r_squre_lg_calibration = [] # r squared of linear regression with calibration data
    r_squre_lg_validation = [] # r squared of linear regression with validation data
    for k in n_factors:
        # index of factors
        # the top k factors combined with bottom k//s factors
        s = k+1
        if k == len(X_df.columns): # the first k include all factors
            index = X_df.columns
        else:
            index = list(X_df.columns[ranking[:k]]) + \
                     list(X_df.columns[ranking[-k//s:]])
        # get X data
        data = X_df.loc[:,index]


        # using pca in LinearRegression
        X = np.array(data)
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
        # record r score
        r_squre_pca_calibration.append(model_pca.score(X_pca,y))
        r_squre_pca_validation.append(model_pca.score(X_test_pca,y_test))
        ## compare with the performance of linear regression without PCA

        model_lg = LinearRegression()# initial predicition model
        model_lg.fit(X,y) # fit model
        # record mse
        mse_lg_calibration.append(mean_squared_error(model_lg.predict(X),y))
        mse_lg_validation.append(mean_squared_error(model_lg.predict(X_test),y_test))
        # record r squre
        r_squre_lg_calibration.append(model_lg.score(X,y)) # r squared of linear regression with calibration data
        r_squre_lg_validation.append(model_lg.score(X_test, y_test)) # r squared of linear regression with validation data
        # present mse
    mse = {'mse_pca_calibration' : mse_pca_calibration, 'mse_pca_validation' : mse_pca_validation, \
                       'mse_lg_calibration' : mse_lg_calibration, 'mse_lg_validation' : mse_lg_validation}
    r_squre = {'r_squre_pca_calibration' : r_squre_pca_calibration,'r_squre_pca_validation' : r_squre_pca_validation,\
                    'r_squre_lg_calibration' : r_squre_lg_calibration, 'r_squre_lg_validation' : r_squre_lg_validation  }
    mse_df = pd.DataFrame(mse , index = np.concatenate([[len(X_df.columns)],np.array(n_factors[1:])  +  np.array(n_factors[1:])//s]))
    r_squre_df = pd.DataFrame(r_squre , index = np.concatenate([[len(X_df.columns)],np.array(n_factors[1:])  +  np.array(n_factors[1:])//s]))
    return mse_df,r_squre_df , model_pca, model_lg


def inital_model(inputs,optimizer = 'adam',loss='mean_squared_error'):
    '''
    construct autoencoder neural network
    output: autoencoder nerual network
    '''
    # input layer
    input_img = Input(shape=(inputs, ))
    # encoding layer ( we need to figure out how many layers we need and how many nodes for each layer)
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_img)#1
    encoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)#3
    encoded = Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)#3
    decoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encoded)#4
    decoded = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(decoded)#5
    decoded = Dense(inputs, activation= 'linear', kernel_regularizer=regularizers.l2(0.01))(decoded) #6
    ######### construct and compile 
    autoencoder = Model(input_img, decoded)
    # dam is the one of most popular one on the market
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder

def get_data(X,Y, P_calibrate ,batch = 10,frac = 0.8 , shuffle = False):
    '''
    store and classify data into calibration and validation; batch shuffle data into different batch;
    input:
        X data and Y data ,
        frac: fraction of data that a batch contain  (0.8 means batch only contain 80% of total data)
        shuffle: whether to shuffle data
    output: defaultdict contains batch samples
    '''
    ori_data = pd.concat([X,Y] ,axis = 1)
    data = ori_data
    # number for calibrate and validation
    N_calibrate = int( P_calibrate * X.shape[0])
    batch_N_calibrate = int(N_calibrate * frac)
    # store and classify data using defaultdict
    X_data = defaultdict(defaultdict)
    Y_data = defaultdict(defaultdict)
    X_data['calibrate']['batch_without_shuffle'] = defaultdict(defaultdict)
    Y_data['calibrate']['batch_without_shuffle'] = defaultdict(defaultdict)
    X_data['validation']['batch_without_shuffle'] = defaultdict(defaultdict)
    Y_data['validation']['batch_without_shuffle'] = defaultdict(defaultdict)
    X_data['calibrate']['batch'] = defaultdict(defaultdict)
    Y_data['calibrate']['batch'] = defaultdict(defaultdict)
    X_data['validation']['batch'] = defaultdict(defaultdict)
    Y_data['validation']['batch'] = defaultdict(defaultdict)
    # store the total df (shuffle == False)
    if shuffle == True:
        data = data.sample(frac = 1)
    X_data['calibrate']['origin']  = data.iloc[:N_calibrate,:-1]
    Y_data['calibrate']['origin']  = data.iloc[:N_calibrate,-1]
    X_data['validation']['origin']  = data.iloc[N_calibrate:,:-1]
    Y_data['validation']['origin']  = data.iloc[N_calibrate:,-1]
    # generate data of returns direction [-1,1]
    # -1 indicate negative returns, 1 indicate positive returns
    Y_data['calibrate']['direction'] = [-1 if i < 0 else 1 for i in Y_data['calibrate']['origin'] ]
    Y_data['validation']['direction'] = [-1 if i < 0 else 1 for i in Y_data['validation']['origin'] ]
    # batch shuffle
    for key in range(batch):
        temp_data = ori_data.sample(frac = frac)
        # calibrate
        X_data['calibrate']['batch']['processed_batch{}'.format(key)] = temp_data.iloc[:batch_N_calibrate,:-1]
        Y_data['calibrate']['batch']['processed_batch{}'.format(key)]  = temp_data.iloc[:batch_N_calibrate,-1]
        # validation
        X_data['validation']['batch']['processed_batch{}'.format(key)]  = temp_data.iloc[batch_N_calibrate:,:-1]
        Y_data['validation']['batch']['processed_batch{}'.format(key)]  = temp_data.iloc[batch_N_calibrate:,-1]
        # generate data of returns direction [-1,1]
        # -1 indicate negative returns, 1 indicate positive returns
        Y_data['calibrate']['batch']['direction_batch{}'.format(key)] = [-1 if i < 0 else 1 for i in Y_data['calibrate']['batch']['processed_batch{}'.format(key)] ]
        Y_data['validation']['batch']['direction_batch{}'.format(key)] = [-1 if i < 0 else 1 for i in Y_data['validation']['batch']['processed_batch{}'.format(key)] ]
        
        start = np.random.choice(int((1 - frac) * data.shape[0]))
        temp_data = ori_data.iloc[start:start + int(frac * data.shape[0]),:]
        # calibrate
        X_data['calibrate']['batch_without_shuffle']['processed_batch{}'.format(key)] = temp_data.iloc[:batch_N_calibrate,:-1]
        Y_data['calibrate']['batch_without_shuffle']['processed_batch{}'.format(key)]  = temp_data.iloc[:batch_N_calibrate,-1]
        # validation
        X_data['validation']['batch_without_shuffle']['processed_batch{}'.format(key)]  = temp_data.iloc[batch_N_calibrate:,:-1]
        Y_data['validation']['batch_without_shuffle']['processed_batch{}'.format(key)]  = temp_data.iloc[batch_N_calibrate:,-1]
        # generate data of returns direction [-1,1]
        # -1 indicate negative returns, 1 indicate positive returns
        Y_data['calibrate']['batch_without_shuffle']['direction_batch{}'.format(key)] = [-1 if i < 0 else 1 for i in Y_data['calibrate']['batch_without_shuffle']['processed_batch{}'.format(key)] ]
        Y_data['validation']['batch_without_shuffle']['direction_batch{}'.format(key)] = [-1 if i < 0 else 1 for i in Y_data['validation']['batch_without_shuffle']['processed_batch{}'.format(key)] ]
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
    '''
    clean the dataframe for na with different method
    output:
        cleaned dataframe
    '''
    if method == 'dropna':
        # dropna and set batch size to 2000
        data = data.dropna()  ;frac = 2000 /data.shape[0]
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

def norml_standard(X_data , m = 'standardlization'):
    '''
    normalization or standardlization
    output: processed dataframe
    '''
    #normalization or standardlization
    if m == 'normalization':
        X_data = pd.DataFrame(normalize(X_data, axis=1) , columns = X_data.columns , index = X_data.index)
    elif m == 'standardlization':
        X_data = pd.DataFrame(StandardScaler().fit_transform(X_data) , columns = X_data.columns , index = X_data.index)
    return X_data


def cut_fat_tail(data, method):
    '''cut the tail to eliminate the fat tail effect'''
    if method == methods[0]:
        right = data.quantile(.995)
        left =  data.quantile(1 - .995)
        data = data[~(data > right).any(axis = 1)]
        data = data[~(data < left).any(axis = 1)]
    else :
        right = data.quantile(.95)
        left =  data.quantile(1 - .95)
        data = data[~(data > right).any(axis = 1)]
        data = data[~(data < left).any(axis = 1)]
    return data

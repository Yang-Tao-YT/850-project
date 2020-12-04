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
import copy
import yfinance
from collections import defaultdict
from os.path import abspath,join
current_folder = abspath('');current_folder
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

def get_data(data, N_calibrate , N_validation):

    data = data
    # number for calibrate and validation
    N_calibrate = int( N_calibrate * data.shape[0])

    # store and classify data using defaultdict
    X_data = defaultdict(defaultdict)
    Y_data = defaultdict(defaultdict)
    # calibrate
    X_data['calibrate']['processed'] = data.iloc[:N_calibrate,list(range(0,10)) + list(range(11,data.shape[1]))]
    Y_data['calibrate']['processed'] = data.iloc[:N_calibrate,10]
    # validation
    X_data['validation']['processed'] = data.iloc[N_calibrate:,list(range(0,10)) + list(range(11,data.shape[1]))]
    Y_data['validation']['processed'] = data.iloc[N_calibrate:,10]   
    
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

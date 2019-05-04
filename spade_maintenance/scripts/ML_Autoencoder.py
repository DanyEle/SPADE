import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#%matplotlib inline

from numpy.random import seed
from tensorflow import set_random_seed

from keras.layers import Input, Dropout
from keras.layers.core import Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json

#Let's define the autoencoder model.



def create_autoencoder(X_train):
    set_random_seed(10)
    act_func = 'elu'

    # Input layer:
    model=Sequential()
    # First hidden layer, connected to input vector X.
    model.add(Dense(10,activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(X_train.shape[1],)
                   )
             )

    model.add(Dense(2,activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(10,activation=act_func,
                    kernel_initializer='glorot_uniform'))

    model.add(Dense(X_train.shape[1],
                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse',optimizer='adam')


    return(model)

    # Train model for 100 epochs, batch size of 10:


def train_model(model, X_train, batch_size, num_epochs, valid_split):
    
    history=model.fit(np.array(X_train),np.array(X_train),
                      batch_size=batch_size,
                      epochs=num_epochs,
                      validation_split=valid_split, #5% of training samples used for validation
                      verbose = 1)
    
    return(model, history)



def show_training_history_loss_plot(history):
    plt.plot(history.history['loss'],
             'b',
             label='Training loss')
    plt.plot(history.history['val_loss'],
             'r',
             label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0, .1])
    plt.show()
    
    
def show_loss_distr_training_set(X_train, model):
    #let's try predicting the values for each training data point
    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred, 
                          columns=X_train.columns)
    X_pred.index = X_train.index
    
    scored = pd.DataFrame(index=X_train.index)
    #Distance of the prediction from the original value
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    plt.figure()
    sns.distplot(scored['Loss_mae'],
                 bins = 10, 
                 kde= True,
                color = 'blue');
    plt.xlim([0.0,.5])
    
    
#The Z-Score is used to understand where the threshold for outliers should lie at.
#ASSUMPTION: The data is normally distributed.!
#By Chebishev's inequality, we know that 99% of 
#the data will lie within 3*STD of the loss' distribution.
#Values outside this distribution are outliers.
#To find out the threshold, we take the mean value of all data points
#that are considered outliers. 
def find_loss_threshold_value(X_train, model, extreme=False):
    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred, columns=X_train.columns)
    X_pred.index = X_train.index
    
    scored = pd.DataFrame(index=X_train.index)
    #Distance of the prediction from the original value
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    
    mean_1 = scored['Loss_mae'].mean()
    std_1 = scored['Loss_mae'].std() 
    
    k = 3 if extreme else 2
    
    outliers=[]
    #Firstly, find all data points that are outliers based on the Z-Test
    #(i.e.: all the values that are outside 3*standard deviation)
    for y in scored['Loss_mae']:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > k:
            outliers.append(y)
            
            
    if(len(outliers) == 0):
        anomaly_threshold=False
    else:
        #Now let's take the mean of all outliers
        anomaly_threshold = np.mean(outliers)

    return(anomaly_threshold)    
    
    
#X_data can be either: X_train or X_test
def mark_data_frame_as_anomaly(X_data, model, threshold_value):
    X_pred = model.predict(np.array(X_data))
    X_pred = pd.DataFrame(X_pred, 
                          columns=X_data.columns)
    X_pred.index = X_data.index
    
    scored = pd.DataFrame(index=X_data.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_data), axis = 1)
    scored['Threshold'] = threshold_value
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    
    return(scored)
    
    
def autoencoder_find_anomaly_threshold(X_data):
    #first thing first, let's initialize the data that we will be needing to generate the autoencoder model
    autoencoder_model = create_autoencoder(X_data)
    trained_model, history = train_model(autoencoder_model, X_data, batch_size=1, num_epochs=100, valid_split=0.2) #0.05
    #Let's see how the loss function evolved during the training process.
    show_training_history_loss_plot(history)   
    #And let's see where a good threshold may lie at by inspecting the loss of the training set. 
    show_loss_distr_training_set(X_data, trained_model)
    #extreme=False --> 95% of data considered. extreme=True --> 99% of data considered.
    anomaly_threshold = find_loss_threshold_value(X_data, trained_model, extreme=False)
    
    return anomaly_threshold, trained_model  
    
    
    
    
    
    
    
    

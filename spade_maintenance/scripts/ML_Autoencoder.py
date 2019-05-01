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


def train_model(model, X_train, batch_size, num_epochs):
    history=model.fit(np.array(X_train),np.array(X_train),
                      batch_size=batch_size,
                      epochs=num_epochs,
                      validation_split=0.05,
                      verbose = 1)


    plt.plot(history.history['loss'],
             'b',
             label='Training loss')
    plt.plot(history.history['val_loss'],
             'r',
             label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0,.1])
    plt.show()


def show_history_loss_plot(history):
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

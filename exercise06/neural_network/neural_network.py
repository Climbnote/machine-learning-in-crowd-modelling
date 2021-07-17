import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


def create_nn(h, K=10, s_k=False, v_rel=False):
    """ creates sequential model for neural network depending on parameters

    param h: list of number of neurons in each hidden layer
    param K: number of nearest neighbors observed
    param s_k: parameter determining if mean Euclidean spacing is used as input feature
    param v_rel: parameter determining if relative velocities are used as input feature
    return: sequential model for created neural network
    """
    # Creating sequential model 
    nn = keras.Sequential()
    # Predefine input size 
    if v_rel == False:
        if s_k == False:
            nn.add(keras.Input(shape=(2*K,)))#NN1
        else:
            nn.add(keras.Input(shape=(2*K+1,)))#NN3
    else: 
        if s_k == False:
            nn.add(keras.Input(shape=(4*K,)))#NN2
        else:
            nn.add(keras.Input(shape=(4*K+1,)))#NN4
    # Add hiddenlayer to the model
    for layer_size in h:
        nn.add(keras.layers.Dense(layer_size, activation="sigmoid"))
    # Add output layer
    nn.add(keras.layers.Dense(1))
    return nn


def train_nn(nn, data_combination, B_train, B_validation, C_train, C_validation, K, batch_size, epochs, number_steps_loss_increase, min_improvement, s_k=False, v_rel=False):
    """ trains neural network

    param nn: sequential model of neural network
    param data_combination: combination of testing/training data applied
    param B_train: Bottleneck training dataset
    param B_validation: Bottleneck validation dataset
    param C_train: Corridor training dataset
    param C_validation: Corridor validation dataset
    param K: number of nearest neighbors observed
    param batch_size: batch size used during training
    param epochs: number of epoch used for training
    param number_steps_loss_increase: number of epochs used for early stopping
    param min_improvement: minimal increment loss used for early stopping
    param s_k: parameter determining if mean Euclidean spacing is used as input feature
    param v_rel: parameter determining if relative velocities are used as input feature
    """
    string = ["x-rel", "y-rel", "v-rel", "u-rel", 'speedInAreaUsingAgentVelocity-PID5']
    if data_combination == 'b':
        data_train = B_train
        data_validation = B_validation
    elif data_combination == 'c':
        data_train = C_train
        data_validation = C_validation
    elif data_combination == 'bc':
        data_train = pd.concat([B_train, C_train]).sample(frac=1, random_state = 1)
        data_validation = pd.concat([B_validation, C_validation]).sample(frac=1, random_state = 1)
    
    data_string = np.array([np.array(["x-rel" + str(i+1), "y-rel" + str(i+1)]) for i in range(K)]).flatten()

    if v_rel == True:
        data_string = np.append(data_string, np.array([np.array(["v-rel" + str(i+1), "u-rel" + str(i+1)]) for i in range(K)]).flatten())

    if s_k == True:
        data_string = np.append(data_string, np.array(['s_k']))
    
    x_train = data_train.loc[:, data_string].to_numpy()
    y_train = data_train['speedInAreaUsingAgentVelocity-PID5'].to_numpy()
    x_val = data_validation.loc[:, data_string].to_numpy()
    y_val = data_validation['speedInAreaUsingAgentVelocity-PID5'].to_numpy()
    xy_val = (x_val, y_val)

    early_stopping = EarlyStopping(monitor='val_loss', patience=number_steps_loss_increase, min_delta=min_improvement, mode='min')

    nn.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=xy_val, callbacks=[early_stopping])


def test_nn(NNi, input_data_idx, NN_id, train_test_sets, B_test, C_test, B_C_test, K):
    """ test neural network models for given parameters
    
    param NNi: list of sequential models of neural network
    param input_data_idx: number determining which neural network is used for training
    param NN_id: number determining which architecture of neural network is used for training
    param train_test_sets:  dataset combination used training and testing
    param B_test: Bottleneck testing dataset
    param C_test: Corridor testing dataset
    param B_C_test: Bottleneck and Corridor combined testing dataset
    param K: number of nearest neighbors observed
    return: list of errors for different tested neural networks
    """
    error_values_NN_id = []
    data_string = np.array([np.array(["x-rel" + str(i+1), "y-rel" + str(i+1)]) for i in range(K)]).flatten()
    if input_data_idx == 1 or input_data_idx == 3:
        data_string = np.append(data_string, np.array([np.array(["v-rel" + str(i+1), "u-rel" + str(i+1)]) for i in range(K)]).flatten())
    if input_data_idx == 2 or input_data_idx == 3:
        data_string = np.append(data_string, np.array(['s_k']))

    for train_set, test_set in train_test_sets:
        set_id = 0
        if train_set == 'B':
            set_id = 0
        elif train_set == 'C':
            set_id = 1
        elif train_set == 'C+B':
            set_id = 2
        if test_set == 'B':
            data = B_test
        elif test_set == 'C':
            data = C_test
        elif test_set == 'C+B':
            data = B_C_test
        x_test = data.loc[:, data_string].to_numpy()
        y_test = data['speedInAreaUsingAgentVelocity-PID5'].to_numpy()
        NN = NNi[input_data_idx][set_id][NN_id]
        results = NN.evaluate(x_test, y_test, verbose=0)
        error_values_NN_id.append(results)
    return error_values_NN_id

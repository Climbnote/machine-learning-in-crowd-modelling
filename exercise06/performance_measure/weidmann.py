import numpy as np
from sklearn.metrics import mean_squared_error


# define parameters for weidmann model of different scenarios
T_corridor = 1 # s
v0_corridor = 1.3 # m/s
l_corridor = 0.4 # m
T_bottleneck = 0.49 # s
v0_bottleneck = 1.7 # m/s
l_bottleneck = 0.4 # m


def calculate_weidmann_speed(sk, v0, T, l):
    """ calculates speed in weidmann model

    param sk: mean Euclidean spacing to K closest neighbors
    param v0: free-walking speed
    param T: time-gap
    param l: pedestrian size
    return: weidmann model predicted speed
    """
    return v0 * (1 - np.exp((l - sk) / (v0 * T)))


def calculate_sk(x_y, xi_yi_array):
    """ calculates mean Euclidean spacing to K closest neighbors of observed pedestrian

    param x_y: x and y values of observed pedestrian
    param xi_yi: x and y values of K closest neighbors to observed pedestrian
    return: calculated mean Euclidean spacing to K closest neighbors
    """
    K = xi_yi_array.shape[0]
    # initialize sk as 0
    sk = 0.
    for xi_yi in xi_yi_array:
        sk += np.sqrt(np.sum(np.square(x_y - xi_yi)))
    return sk / K


def calculate_corridor_speed(sk):
    """ predicts speed in corridor scenario using weidmann model
    
    param sk: mean Euclidean specing
    return: speed
    """
    return calculate_weidmann_speed(sk, v0_corridor, T_corridor, l_corridor)


def calculate_bottleneck_speed(sk):
    """ predicts speed in bottleneck scenario using weidmann model
    
    param sk: mean Euclidean specing
    return: speed
    """ 
    return calculate_weidmann_speed(sk, v0_bottleneck, T_bottleneck, l_bottleneck)


def calculate_b_and_c_speed(sk): 
    """ predicts speed in corridor and bottleneck scenarios combined using weidmann model
    
    param sk: mean Euclidean specing
    return: speed
    """
    T = (T_bottleneck + T_corridor) / 2 # s
    v0 = (v0_bottleneck + v0_corridor) / 2 # m/s
    l = (l_bottleneck + l_corridor) / 2 # m
    return calculate_weidmann_speed(sk, v0, T, l)


def test_weidmann(train_test_sets, B_test, C_test, B_C_test):
    """ calculates mean squared error of weidmann model prediction and real speed values

    param train_test_sets: dataset used for setting parameters of weidmann model and testing them
    param B_test: Bottleneck scenario testing dataset
    param C_test: Corridor scenario testing dataset
    param B_C_test: Bottleneck and Corridor scenarios combines dataset
    return: error obtained during testing
    """
    error_values_weidmann = []
    for train_set, test_set in train_test_sets:
        s_k = np.array(['s_k'])
        if train_set == 'B':
            fun = calculate_bottleneck_speed
        elif train_set == 'C':
            fun = calculate_corridor_speed
        elif train_set == 'C+B':
            fun = calculate_b_and_c_speed
        if test_set == 'B':
            data = B_test
        elif test_set == 'C':
            data = C_test
        elif test_set == 'C+B':
            data = B_C_test
        y_prediction = fun(data.loc[:, s_k].to_numpy()) # apply weidmann model
        y_test = data['speedInAreaUsingAgentVelocity-PID5'].to_numpy()
        results = mean_squared_error(y_prediction, y_test)
        error_values_weidmann.append(results)
    return error_values_weidmann

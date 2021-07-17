import numpy as np


def aic_total(optimal_indices, n, training_set, NNi, error_values_testing, error_values_weidmann):

    aic_tot = []

    for t in range(len(training_set)):
        ks = []
        MSE_fun = []
        aic_fun = []

        # Compute number of paramters for neural networks and 
        for i, nn in enumerate(optimal_indices):
            ks.append(get_number_params(NNi[i][training_set[t]][nn]))
            MSE_fun.append(error_values_testing[i][nn][t])
            aic_fun.append(AIC(ks[i], n, MSE_fun[i]))
        
        aic_weidmann = AIC(3, n, error_values_weidmann[t])
        aic_diff = aic_weidmann - aic_fun
        aic_tot.append(aic_diff)

    aic_values = np.array(aic_tot).T
    return aic_values


def get_number_params(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    return trainableParams


def AIC(k, n, MSE):
    # k: number of parameters
    # n: number of observations
    # mse: mean squared error
    aic = 2*k + n*np.log(MSE) + n*(1 + np.log(2*np.pi))
    return aic
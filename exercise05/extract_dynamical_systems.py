import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def finite_difference(x_0, x_1, delta_t):
    """finite difference formula to approximate vectors v at all points x_0

    :param x_0: data points x_0
    :type x_0: two dimensional numpy array with shape (1000, 2)
    :param x_1: data points x_1
    :type x_1: two dimensional numpy array with shape (1000, 2)
    :param delta_t: the time step size
    :type delta_t: float
    :return: vectors for each datapoint x_0
    :rtype: two dimensional numpy array with shape (1000, 2)
    """
    return (x_1 - x_0) / delta_t


def least_squares_minimization(X, Y, rcond):
    """returns the least-squares solution to a linear matrix equation using numpy.linalg.lstsq

    :param X: the coefficient matrix
    :type X: (M, N) array_like
    :param Y: the ordinate values
    :type Y: {(M,), (M, K)} array_like
    :return: least squares solution
    :rtype: {(N, K), (N,)} ndarray
    """
    return lstsq(X, Y, rcond=rcond)[0].T


def right_hand_side(x, t, A):
    """definition of the right hand side of the linear system of equations to solve

    :param x: the vector of the state variables x1 and x2
    :param t: time
    :param A: the 2x2 matrix with the linear coefficients
    :return: the right hand side vector of the linear system
    """
    x1, x2 = x
    f = [A[0][0] * x1 + A[0][1] * x2,
         A[1][0] * x1 + A[1][1] * x2]
    return f


def solve_linear_system(A, start_points, t_0, t_end, resolution):
    """solves the linear system defined by the matrix A from t_0 to t_end with a defined resolution using scipy.integrate.odeint
        multiple start points can be defined, for each start point only the last solution is saved resulting in vector of two columns

    :param A: the matrix defining the linear system
    :param start_points: x0
    :param t_0: start time
    :param t_end: end time
    :param resolution: resolution of the time vector
    :return: a vstack of the solution of every start point at t_end
    """
    x1_hat = np.empty((0,2))
    for data_point in start_points:
        time = np.arange(t_0,t_end + t_end/resolution,t_end/resolution)
        sol = odeint(right_hand_side, data_point, time, args=(A,))
        #save only last solution
        x1_hat = np.vstack([x1_hat, sol[-1]])
    return x1_hat


def solve_linear_system_full_data(A, start_point, t_0, t_end, resolution):
    """solves the linear system defined by the matrix A from t_0 to t_end with a defined resolution using scipy.integrate.odeint
        one start point can be defined, this time all time steps of the solution are saved

    :param A: the matrix defining the linear system
    :param start_point: x0
    :param t_0: start time
    :param t_end: end time
    :param resolution: resolution of the time vector
    :return: the solution over time for one start point
    """
    time = np.arange(t_0,t_end + t_end/resolution,t_end/resolution)
    return odeint(right_hand_side, start_point, time, args=(A,))


def calculate_x_from_vector(v, x0, delta_t):
    """calculates data value with given flow in current step, data values in previous step and time change defined 

    :param v: vector representing flow for next timestep
    :param x0: previous data space
    :param delta_t: time change defined
    :return: data space for new state calculated
    """
    return delta_t*v + x0


def plot_phase_diagram(projected_data, projected_vectors):
    """plots phase diagram for diven data and derivatives using plt.quiver function

    :param projected_data: array of data values in different number of timesteps
    :param projected_vectors: array of derivative values of projected_data
    :return: plot of phase diagram with first pedestrian movement marked in red
    """
    for i in range(len(projected_data)):
        plt.quiver(projected_data[i][:, 0], projected_data[i][:, 1], projected_vectors[i][:, 0], projected_vectors[i][:, 1], color='black', zorder=1)
        plt.quiver(projected_data[i][100][0], projected_data[i][100][1], projected_vectors[i][100][0], projected_vectors[i][100][1], color='red', zorder=2)


def calculate_new_data_and_derivatives_linear(timesteps, A, data_x0, delta_t):
    """calculates vector field and data approximation for number of timesteps defined in timesteps parameter, when applying linear
    vector field approximation

    :param timesteps: number of timesteps for which calculation should be executed
    :param A: matrix A required for linear vector field approximation
    :param data_x0: data matrix for the first timestep
    :param delta_t: time change defined
    :return: projected data and projected derivative values for timesteps(parameter) number of steps
    """
    projected_data = []
    projected_vectors = []
    projected_data.append(data_x0)
    for i in range(timesteps):
        vector = projected_data[i].dot(A)
        new_x = calculate_x_from_vector(vector, projected_data[i], delta_t)
        projected_vectors.append(vector)
        projected_data.append(new_x)
    projected_vectors.append(np.zeros((2000, 2)))
    return projected_data, projected_vectors


def calculate_new_data_and_derivatives_rbf(timesteps, C, data_x0, delta_t, phi):
    """calculates vector field and data approximation for number of timesteps defined in timesteps parameter, when applying nonlinear
    vector field approximation

    :param timesteps: number of timesteps for which calculation should be executed
    :param C: matrix C required for nonlinear vector field approximation
    :param data_x0: data matrix for the first timestep
    :param delta_t: time change defined
    :param resolution: resolution of the time vector
    :return: projected data and projected derivative values for timesteps(parameter) number of steps
    """
    projected_data = []
    projected_vectors = []
    projected_data.append(data_x0)

    for i in range(timesteps):
        vector = phi(projected_data[i]).dot(C)
        new_x = calculate_x_from_vector(vector, projected_data[i], delta_t)
        projected_vectors.append(vector)
        projected_data.append(new_x)
    projected_vectors.append(np.zeros((2000, 2)))
    return projected_data, projected_vectors


def rbf(x, xl, epsilon):
    """calculates radial basis function with given x, mean and variance epsilon

    :param x: the matrix for which rbf is calculated
    :param xl: array defining means for radial basis functions
    :param epsilon: defines variance for each radial basis function
    :return: the solution of radial basis function
    """
    r = np.exp(-(x-xl)**2 / (epsilon**2))
    return r


def find_radial_basis_function_means(X, number_of_centers):
    """calculates matrix which defines radial basis function means used for regression, uniformly distributed 
    between minimum and maximum values of points in each dimension of matrix X

    :param X: data of system
    :param number_of_centers: number of means which we need
    :return: array of mean points corresponding to X dimension
    """
    dim_mins = []
    dim_maxs = []
    all_ranges = []
    range_matrix = np.empty((0, number_of_centers))
    for i in range(X.shape[1]):
        dim_mins.append(np.min(X[:, i]))
        dim_maxs.append(np.max(X[:, i]))
        all_ranges.append(np.arange(dim_mins[i], dim_maxs[i], (dim_maxs[i]-dim_mins[i])/number_of_centers))
        range_matrix = np.vstack((range_matrix, all_ranges[i]))
    return range_matrix.T


def phi(X, radial_means, epsilon):
    """applys phi function on matrix X and returns trasformed matrix in regard to defined rbf function

    :param X: data matrix
    :param radial_means: mean points for rbf functions
    :param epsilon: variance parameter used in rbf function
    :return: transformed matrix A
    """
    phi_X = np.empty((X.shape[0], 0)) 
    for el in radial_means:
        new_X = rbf(X, el, epsilon)
        phi_X = np.hstack((phi_X, new_X))
    return phi_X


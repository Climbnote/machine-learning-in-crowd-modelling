import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

def plot_2d_dataset(dataset, c=None, cmap=None, scatter=True):
    """plots a two-dimensional dataset with either scatter or plot

    :param dataset: the data set to be plottet
    :type dataset: 2d numpy array with first colummn for x and second column for y
    :param scatter: whether to use scatter or not, defaults to True
    :type scatter: bool, optional
    :param c: the univariate position of the sample according to the main dimension of the points
    :type c: ndarray of shape (dataset.shape,)
    :param cmap: color map name
    :type cmap: string
    :return: the plot
    :rtype: matplotlib.pyplot
    """
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    if scatter:
        ax.scatter(dataset[:,0], dataset[:,1], c=c, cmap=cmap)
    else:
        ax.plot(dataset[:,0], dataset[:,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return plt

def plot_3d_dataset(dataset, c=None, cmap=None):
    """plots a three-dimensional dataset with either scatter or plot

    :param dataset: the data set to be plottet
    :type dataset: 2d numpy array with first column for x, second column for y, and third column for z
    :param c: the univariate position of the sample according to the main dimension of the points in the manifold
    :type c: ndarray of shape (dataset.shape,)
    :param cmap: color map name
    :type cmap: string
    :return: the plot
    :rtype: matplotlib.pyplot
    """
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=c, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=70)
    plt.tight_layout()
    return plt

def plot_eigenvectors_pairwise(eigenvectors, c, L, scatter=True, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1)):
    """scatter plot to plot the second eigenvector on the x-axis and the remaining eigenvectors on the y-axis

    :param eigenvectors: eigenvectors of the kernel matrix which a sorted in ascending order
    :param c: distinct color points for visualizing
    :param L: the number of subplots to plot
    :param xlim: limits of x-axis, defaults to (-0.1, 0.1)
    :type xlim: tuple, optional
    :param ylim: limits of y-axis, defaults to (-0.1, 0.1)
    :type ylim: tuple, optional
    :return: the plot
    :rtype: matplotlib.pyplot
    """
    #prepare data and create copy of eigenvectors without the first non constant eigenvector phi_1
    mask = [i for i in range(np.shape(eigenvectors)[1]) if i != 1]
    eigenvectors_masked = eigenvectors[:,mask]

    #plot first non-constant eigenfunctions phi_1 against the other eigenfunctions in 2D plots in 5 columns and rows dependent on L
    figsize = 3
    numColumns = 5
    numRows = int(math.ceil(L / numColumns))
    fig, axes = plt.subplots(numRows, numColumns, figsize=(numColumns*figsize,numRows*figsize), sharey=True, sharex=True)

    #create copy of eigenfunctions without first non-constant eigenfunction phi_1
    for i in range(numRows):
        for j in range(numColumns):
            if (i*numColumns + j) < L:
                if scatter:
                    axes[i][j].scatter(eigenvectors[:,1], eigenvectors_masked[:,(i*numColumns + j)], c=c, cmap='Spectral')
                else:
                    axes[i][j].plot(eigenvectors[:,1], eigenvectors_masked[:,(i*numColumns + j)])
                idx = i*5 + j + 1
                if j == 0 and i == 0:
                    idx = 0
                tick_spacing = 0.05
                axes[i][j].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                axes[i][j].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                axes[i][j].set_ylabel(r'$\phi_{' + str(idx) + '}$')
                axes[i][j].set_xlabel(r'$\phi_{1}$')
                axes[i][j].set_xlim(xlim[0], xlim[1])
                axes[i][j].set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    return plt

from automaton import CellState
from automaton import Parameters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation


def make_color_map():
    """helper function which creates a custom color map with distinct colors for the states of the cellular automaton

    :return: the color map
    :rtype: colors.ListedColorMap
    """
    #make a color map of fixed colors
    cmap = colors.ListedColormap(['white', 'orange', 'red', 'green'])
    bounds = [CellState.E, CellState.P, CellState.O, CellState.T]
    colors.BoundaryNorm(bounds, cmap.N)
    return cmap


def add_state_text_to_axis(data, ax):
    """helper function which adds labels to cells for the states of the cellular automaton

    :param data: the 2d array which represents the states of the cellular automaton
    :type data: numpy.array
    :param ax: the axes obejct which to edit
    :type ax: matplotlib.axes
    """
    #print the states of the state space of the cellular automata ('P' for pedestrian, 'O' for obstacle and 'T' for target)
    ax.texts.clear()
    for (i, j), state in np.ndenumerate(data):
        #print(i, j)
        if state == CellState.P.value:
            ax.text(j, i, CellState.P.name, ha='center', va='center')
        if state == CellState.O.value:
            ax.text(j, i, CellState.O.name, ha='center', va='center')
        if state == CellState.T.value:
            ax.text(j, i, CellState.T.name, ha='center', va='center')


def add_grid_and_remove_ticks(data, ax):
    """helper function which adds a grid to an axes object and removes all ticks and labels

    :param data: the 2d array which represents the states of the cellular automaton 
    :type data: numpy.array
    :param ax: the axes obejct which to be edited
    :type ax: matplotlib.axes
    """
    #disable all ticks and axis labels
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        )

    #add grid lines
    ax.set_xticks(np.arange(-.5, data.shape[1], 1))
    ax.set_yticks(np.arange(-.5, data.shape[0], 1))
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)


def plot_automaton_states(states):
    """plots the current states of the cellular automaton as a square grid using matshow

    :param states: the 2d array which represents the states of the cellular automaton 
    :type states: numpy.array
    :return: the resulting figure
    :rtype: matplotlib.figure
    """
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    #use matshow wtih custom color map to visualize the cellular automaton
    ax.matshow(states, cmap=make_color_map())
    #set state texts to cells
    add_state_text_to_axis(states, ax)
    #remove labels and ticks and add grid
    add_grid_and_remove_ticks(states, ax)
    #return plot
    return fig


def plot_automaton_simulation(data, parameters = Parameters()):
    """plots an animation of different states of the cellular automaton as a square grid using matshow

    :param data: a list of 2d arrays which represent the states of the cellular automaton for each time step
    :type data: list
    :param interval: determines how long each frame should be shown in millisec
    :type interval: int
    :return: the resulting animation
    :rtype: animation.FuncAnimation
    """
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    #use matshow wtih custom color map to visualize the cellular automaton
    plot = plt.matshow(data[0], fignum=0, cmap=make_color_map())
    add_grid_and_remove_ticks(data[0], ax)

    def init():
        plot.set_data(data[0])
        add_state_text_to_axis(data[0], ax)
        return plot

    def update(i):
        plot.set_data(data[i])
        add_state_text_to_axis(data[i], ax)
        return [plot]

    #calculate time per simulation frame in milli sec
    interval = 1000 * (parameters.cell_width / parameters.speed_pedestrian)
    #create animation
    return animation.FuncAnimation(fig, update, init_func=init, frames=len(data), interval=interval, repeat=False)

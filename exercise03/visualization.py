import numpy as np
import matplotlib.pyplot as plt
import sir_model

def bifurcation_diagram_2d(f,f_x,alpha_lim=(-1,1)):
    """plots the bifurcation diagram for any 1d function f and its derivative f_x indicating stable and unstable equilibria

    :param f: function
    :param f_x: derivative
    :param alpha_lim: x limits of plot, defaults to (-1,1)
    :type alpha_lim: tuple, optional
    :return: plot
    :rtype: matplotlib plot
    """
    x_lim = (-1, 1)
    alpha_lim = alpha_lim

    #create mesh grid
    resolution = 100
    x, alpha = np.meshgrid(np.linspace(x_lim[0], x_lim[1], resolution), np.linspace(alpha_lim[0], alpha_lim[1], resolution))
    
    #extract the 0 level set of f
    plt.subplots(figsize=(6,6))
    CS = plt.contour(alpha,x,f(alpha,x),[0],colors='k')
    #clear figure
    plt.clf()
    c0 = CS.collections[0]
    # for each path in the contour extract vertices and mask by the sign of df/dx
    for path in c0.get_paths():
        vertices = path.vertices
        v_alpha = vertices[:,0]
        v_x = vertices[:,1]
        mask = np.sign(f_x(v_alpha,v_x))
        stable = mask < 0.
        unstable = mask > 0.
        
        # plot the stable and unstable branches for each path
        plt.plot(v_alpha[stable],v_x[stable],'b')
        plt.plot(v_alpha[unstable],v_x[unstable],'b--')
        
    plt.xlabel("α")
    plt.ylabel('x')
    plt.legend(('stable','unstable'),loc='best')
    plt.xlim(alpha_lim[0],alpha_lim[-1])
    plt.ylim(x_lim[0],x_lim[-1])

    #return plot
    return plt


def phase_diagram_with_trajectory_2d(f,alpha,starts,arrows=True):
    """plots a 2d phase portrait for any given function f parameterized with one parameter alpha using either quiver or plot and optional additional trajectories

    :param f: function
    :param alpha: parameter 
    :type alpha: float
    :param starts: optional start values for trajectories
    :type starts: numpy array
    :param arrows: boolean whether to use quiver instead of streamplot or not, defaults to True
    :type arrows: bool, optional
    :return: plot
    :rtype: matpllotlib plot
    """
    x_lim = (-1, 1)
    y_lim = (-1, 1)

    #two different resolutions, one for the vector field using quiver / streamplot, and one for the trajectory using streamplot
    resolution_traj = 200
    resolution_vec = 15

    x_traj, y_traj = np.meshgrid(np.linspace(x_lim[0], x_lim[1], resolution_traj), np.linspace(y_lim[0], y_lim[1], resolution_traj))
    x_vec, y_vec = np.meshgrid(np.linspace(x_lim[0], x_lim[1], resolution_vec), np.linspace(y_lim[0], y_lim[1], resolution_vec))

    #create figure
    fig, ax = plt.subplots(figsize=(6,6))

    #plot vector field with quiver or streamplot
    if arrows:
        ax.quiver(x_vec, y_vec, f(x_vec, y_vec, alpha)[0], f(x_vec, y_vec, alpha)[1], color="black")
    else:
        ax.streamplot(x_vec, y_vec, f(x_vec, y_vec, alpha)[0], f(x_vec, y_vec, alpha)[1], color="black")

    #plot trajectory from defined starting point
    ax.streamplot(x_traj, y_traj, f(x_traj, y_traj, alpha)[0], f(x_traj, y_traj, alpha)[1], start_points=starts, color="red", linewidth=3, arrowsize=3)
    
    plt.xlabel("x")
    plt.ylabel('y')

    #return plot
    return plt


def sir_2d(sol, ax):
    """plots the number of S, I and R persons vs time

    :param sol: [description]
    :type sol: [type]
    :param ax: [description]
    :type ax: [type]
    :return: [description]
    :rtype: [type]
    """
    ax.plot(sol.t, sol.y[0], label='1E0*susceptible')
    ax.plot(sol.t, 1e3*sol.y[1], label='1E3*infective')
    ax.plot(sol.t, 1e1*sol.y[2], label='1E1*removed')
    ax.set_xlim([0, 500])
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(r"$S,I,R$")
    return ax


def recovery_rate(sol, param, ax):
    """plots the recovery rate and the min and max value as boundary horizontal lines

    :param sol: solution of numerical integration obtaining the number of S, I and R 
    :param param: parameter object of SIR model
    :type param: Parameters
    :param ax: axes object to plot on
    :type ax: matplotlib.axes
    """
    ax.plot(sol.t, sir_model.mu(param.b, sol.y[1], param.mu0, param.mu1), label='recovery rate')
    ax.axhline(y=param.mu0, color='r', linestyle='dotted', label='$\mu_0$')
    ax.axhline(y=param.mu1, color='r', linestyle='dashed', label='$\mu_1$')
    ax.set_xlim([0, 500])
    ax.set_ylim([param.mu0 - 0.05, param.mu1 + 0.05])
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mu$")
    ax.legend()
    return ax


def indicator_function(I_h, param, ax):
    """plot the indicator function for hopf bifurcations

    :param I_h: array of I values
    :param param: parameter object of SIR model
    :type param: Parameters
    :param ax: axes object to plot on
    :type ax: matplotlib.axes
    """
    ax.plot(I_h, sir_model.h(I_h, param.mu0, param.mu1, param.beta, param.A, param.delta, param.nu, param.b))
    ax.plot(I_h, 0*I_h, 'b:')
    ax.set_xlabel("I")
    ax.set_ylabel("h(I)")
    return ax


def sir_3d(sol, time, ax, resolution, color):
    """plots the 3d representation of the SIR model solution

    :param sol: solution of numerical integration obtaining the number of S, I and R
    :param time: timesteps of integration
    :param ax: axes object to plot on
    :param resolution: plot every resolution'th point
    :type resolution: int
    :param color: color of data
    """
    n = resolution
    ax.scatter(sol.y[0][::n], sol.y[1][::n], sol.y[2][::n], s=1, color=color)
    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")
    ax.set_xlim(192,200)
    ax.set_ylim(0.00,0.09)
    ax.set_zlim(1,7)
    return ax


def sir_2d_projection(sol, time, ax, resolution, color):
    """plots the 2d projection of the SIR model onto the SI plane

    :param sol: solution of numerical integration obtaining the number of S, I and R
    :param time: timesteps of integration
    :param ax: axes object to plot on
    :param resolution: plot every resolution'th point
    :type resolution: int
    :param color: color of data
    """
    n = resolution
    ax.scatter(sol.y[0][::n], sol.y[1][::n], s=1, color=color)
    ax.set_xlabel("S")
    ax.set_ylabel("I")
    return ax
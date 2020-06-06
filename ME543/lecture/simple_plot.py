import matplotlib.pyplot as plt
import numpy as np

def arrayplot(x, y, labels=['x','y'], titlestring='', axis_range='automatic', linestyle='solid', linewidth=3, marker='', grid=False):
    """
    produce a plot based on a 1D array of abscissas and a 1D or 2D array of ordinates
    
    inputs:
    x: numpy array of abscissa values
    y: numpy array of ordinate values
    labels: a list of strings for labeling the axes
    titlestring: a string specifying the title of the plot
    axis_range: 'automatic' or list with 4 entries x_min, x_max, y_min, y_max
    -----
    product: 2D plot of the data
    """
    plt.plot(x,y, linestyle=linestyle, linewidth=linewidth, marker=marker)
    plt.xlabel('x')
    plt.ylabel('y')
    if 'automatic' == axis_range:
        axis_range = np.array([x[0],x[-1],np.min(y),np.max(y)])
    plt.axis(axis_range)
    plt.title(titlestring)
    plt.grid(grid)
    plt.show()
    
def indexplot(x, y, indices='all', labels=['x','y'], titlestring='', axis_range='automatic', linestyle='solid', linewidth=3, marker=''):
    """
    produce a plot based on a 1D array of abscissas and select indices of a 2D array of ordinates
    
    inputs:
    x: numpy array of abscissa values
    y: 2D numpy array of ordinate values for multiple plots
    indices: list of indices to be plotted
    labels: a list of strings for labeling the axes
    titlestring: a string specifying the title of the plot
    axis_range: 'automatic' or list with 4 entries x_min, x_max, y_min, y_max
    -----
    product: 2D plot of the data
    """
    n = y.shape[1]
    ymin, ymax = -1, 1
    if 'all'==indices:
        indices = range(n)
    for i in indices:
        ymin = min(ymin, np.min(y.T[i]))
        ymax = max(ymax, np.max(y.T[i]))
        plt.plot(x,y.T[i], linestyle=linestyle, linewidth=linewidth, marker=marker)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if 'automatic' == axis_range:
        axis_range = np.array([x[0],x[-1],ymin,ymax])
    plt.axis(axis_range)
    plt.title(titlestring)
    plt.show()
    
def arraycontourplot(xvals, yvals, fvals, levels=[-1000,0], labels=['x','y'], 
    titlestring='', filled=False):
    """
    inputs:
    xvals: a 1d numpy array of values for the first coordinate
    yvals: a 1d numpy array of values for the second coordinate
    fvals: a 2d numpy array of function values on the coordinate grid
    levels: a list of contour values
    vars: a list containing the symbolic variables
    titlestring: a string specifying the plot title
    -----
    product: a contourplot based on the array of function values
    """
    fig = plt.figure()
    X,Y = np.meshgrid(yvals,xvals) #switch for more intuitive format
    if filled==True:
        cp = plt.contourf(X, Y, fvals, levels, hatches=['x','+']) #, linestyles='dashed')
    else:
        cp = plt.contour(X, Y, fvals, levels) #, linestyles='dashed')
    # plt.clabel(cp, inline=True, fontsize=10)
    plt.title(titlestring)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    #plt.axis('square')
    plt.axis('tight')
    plt.show()
    return cp
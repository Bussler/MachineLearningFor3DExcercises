"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with positive values outside the shape and negative values inside.
    """

    array = np.zeros((resolution, resolution, resolution))
    x = np.empty(resolution* resolution* resolution)
    y = np.empty(resolution* resolution* resolution)
    z = np.empty(resolution* resolution* resolution)
    factor = 1 / resolution

    counter = 0

    for i in range(0,resolution):
        for j in range(0,resolution):
            for k in range(0,resolution):
                x[counter] = (i*factor) - 0.5
                y[counter] = (j*factor) - 0.5
                z[counter] = (k*factor) - 0.5
                counter += 1

    sdfValues = sdf_function(x, y, z)
    array = sdfValues.reshape((resolution, resolution, resolution))

    return array

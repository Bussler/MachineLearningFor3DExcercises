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

    # ###############
    # TODO: Implement
    array = np.zeros((resolution, resolution, resolution))

    factor = 1 / resolution

    for i in range(0,resolution):
        for j in range(0,resolution):
            for k in range(0,resolution):
                x = 0.5 - (i*factor)
                y = 0.5 - (j*factor)
                z = 0.5 - (k*factor)
                array[i,j,k] = sdf_function(np.array([x]),np.array([y]),np.array([z]))

    return array
    # ###############

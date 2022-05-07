"""Definitions for Signed Distance Fields"""
from cmath import sqrt
import numpy as np


def signed_distance_sphere(x, y, z, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a sphere of radius r, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param r: radius of the sphere
    :param x_0: x coordinate of the center of the sphere
    :param y_0: y coordinate of the center of the sphere
    :param z_0: z coordinate of the center of the sphere
    :return: signed distance from the surface of the sphere
    """
    # ###############
    # TODO: Implement
    helper = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)
    helper = np.sqrt(helper)
    return helper - r
    # ###############


def signed_distance_torus(x, y, z, R, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a torus of minor radius r and major radius R, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param R: major radius of the torus
    :param r: minor radius of the torus
    :param x_0: x coordinate of the center of the torus
    :param y_0: y coordinate of the center of the torus
    :param z_0: z coordinate of the center of the torus
    :return: signed distance from the surface of the torus
    """
    # ###############
    # TODO: Implement
    a = np.sqrt(np.power(x, 2) + np.power(z, 2)) - R # TODO use helper again for rounding?
    return np.sqrt(np.power(a, 2) + np.power(y, 2)) - r
    # ###############


def signed_distance_atom(x, y, z):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a hydrogen atom consisting of a spherical proton, a torus orbit, and one spherical electron
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :return: signed distance from the surface of the hydrogen atom
    """
    proton_center = (0, 0, 0)
    proton_radius = 0.1
    orbit_radius = 0.35  # The major radius of the orbit torus
    orbit_thickness = 0.01  # The minor radius of the orbit torus
    electron_center = (orbit_radius, 0, 0)
    electron_radius = 0.05
    # ###############
    # TODO: Implement

    distProton = signed_distance_sphere(x,y,z, proton_radius, proton_center[0], proton_center[1], proton_center[2])
    distElectron = signed_distance_sphere(x, y, z, electron_radius, electron_center[0], electron_center[1], electron_center[2])
    distOrbit = signed_distance_torus(x, y, z, orbit_radius, orbit_thickness, proton_center[0], proton_center[1], proton_center[2])

    #concatDistances = np.concatenate((distProton, distElectron, distOrbit)).reshape(3, len(x))
    #minIndices = np.argmin(np.abs(concatDistances), axis = 0)

    #minVal = np.zeros(len(x))
    #for i in range(0, len(x)):
    #    minVal[i] = concatDistances[minIndices[i], i]

    #return minVal
    #return np.minimum(np.minimum(distProton, distOrbit), distElectron) # TODO correct? -> Absolute minimum?
    concatDistances = np.array([distProton, distOrbit, distElectron])
    return np.amin(concatDistances, axis=0)
    # ###############

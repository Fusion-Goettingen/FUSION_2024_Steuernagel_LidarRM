"""
Contains general utility functions.
"""
import numpy as np
from shapely.geometry import Polygon


def rot(theta):
    """
    Constructs a rotation matrix for given angle alpha.
    :param theta: angle of orientation
    :return: Rotation matrix in 2D around theta (2x2)
    """
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return r.reshape((2, 2))


def pol2cart_pt(point):
    """Given a point as 2 values in polar coordinates, return the Cartesian representation"""
    return pol2cart(*point)


def cart2pol_pt(point):
    """Given a point as 2 values in Cartesian coordinates, return the polar representation"""
    return cart2pol(*point)


def pol2cart(rho, phi):
    """
    Convert polar coordinates to Cartesian. Distance and angle are given as separate parameters.
    :param rho: Distance to origin
    :param phi: Angle to x-Axis
    :return: [x, y] in Cartesian Coordinates as numpy array
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar. X/Y are given as separate parameters
    :param x: x location
    :param y: y location
    :return: ndarray of [distance, angle]
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return np.array([rho, phi])


def identity(x):
    """Helper function equivalent to lambda x: x"""
    return x


def matrix_to_params(X):
    """Convert shape matrix X to parameter form [alpha, l1, l2] with semi-axis length"""
    assert X.shape == (2, 2), "X is not a 2x2 matrix"
    val, vec = np.linalg.eig(X)  # eigenvalue decomposition
    alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
    alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
    p = [alpha, *np.sqrt(val)]
    return np.array(p)


def params_to_matrix(p):
    """
    Convert parameters [alpha, l1, l2] to shape matrix X (2x2)
    """
    X = rot(p[0]) @ np.diag(np.array(p[1:]) ** 2) @ rot(p[0]).T
    return X


def state_to_rect_corner_pts(m, p):
    """
    Given a state represent by kinematic mean and shape state, return the corner locations of the corresponding
    rectangles. The first corner is returned twice, so that a plot of the 5 value pairs will be "closed".
    :param m: 2D kinematic state
    :param p: orientation, major semi-axis, minor semi-axis
    :return: corners of the rectangle, with the first corner repeated at the end, as 5x2 ndarray
    """
    theta, l, w = p
    pts = np.array([
        [l, w],
        [l, -w],
        [-l, -w],
        [-l, w],
        [l, w]
    ])
    rmat = rot(theta)
    pts = [rmat @ p + m for p in pts]
    return np.array(pts)


def state_to_ellipse_contour_pts(m, p, n_pts=100):
    ellipse_angle_array = np.linspace(0.0, 2.0 * np.pi, n_pts)
    pts = (m[:, None] + rot(p[0]) @ np.diag([p[1], p[2]]) @ np.array(
        [np.cos(ellipse_angle_array), np.sin(ellipse_angle_array)])).T
    return np.array(pts)


def get_shapely_rectangle(m, p) -> Polygon:
    """Get shapely.Polygon from semi-axis representation"""
    return Polygon(state_to_rect_corner_pts(m, p))

import numpy as np
from scipy.spatial.distance import cdist


def get_max_angular_distance_pair(measurements):
    """
    Given a set of measurements in Cartesian coordinates, return the two indices of the point pair with the
    maximum distance between them, where the distance is evaluated in radians
    :param measurements: Nx2 Cartesian points
    :return: ix0, ix1: indices of relevant pair
    """
    angles = np.arctan2(measurements[:, 1], measurements[:, 0]) % (2 * np.pi)
    unit_circle_cartesians = np.zeros_like(measurements)
    unit_circle_cartesians[:, 0] = np.cos(angles)
    unit_circle_cartesians[:, 1] = np.sin(angles)
    distance_matrix = cdist(unit_circle_cartesians, unit_circle_cartesians)
    distance_matrix[np.tril_indices_from(distance_matrix)] = 0  # only look at upper triangular for max
    ix0, ix1 = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

    return ix0, ix1

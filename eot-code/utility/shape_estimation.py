import numpy as np
from utility.utils import matrix_to_params


def scatter_matrix(Z, m=None, normalize=False):
    normalization = len(Z) if normalize else 1
    if m is None:
        m = np.average(Z, axis=0)
    m = m.reshape((-1, 2))
    Z_bar = Z - m
    Z_bar = (Z_bar.T @ Z_bar) / normalization
    return Z_bar


def mean_shifted_scattering_matrix_shape_estimate(Z, m=None, normalize=False, scaling_factor=0.25):
    if m is None:
        m = np.average(Z, axis=0)
    Z_bar = scatter_matrix(Z, m=m, normalize=normalize)
    p = matrix_to_params(Z_bar/scaling_factor)
    return p

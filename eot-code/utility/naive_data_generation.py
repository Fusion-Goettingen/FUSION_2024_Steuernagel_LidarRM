import numpy as np

from utility.utils import rot


def get_rectangular_measurements(loc, length, width, theta, n_measurements, R, weight_list=None, internal_RNG=None):
    if weight_list is None:
        weight_list = np.array([0.25, 0.25, 0.25, 0.25])
    # make sure weight_list sums to 1
    weight_list = np.array(weight_list) / np.sum(weight_list) if np.sum(weight_list) != 1 else np.array(
        weight_list)

    if n_measurements < 1:
        n_measurements = 1

    if internal_RNG is None:
        internal_RNG = np.random.default_rng()

    # half axis length for quick reference
    half_length = length / 2
    half_width = width / 2

    # measurements Y
    Y = []

    while len(Y) < n_measurements:
        # decide side based on weight_list
        #   0 3 3 3 3 3 2
        #   0     c     2
        #   0 1 1 1 1 1 2
        # where c is the object center, sides 0/2 are the width, and 1/3 are the length
        side_ix = internal_RNG.choice([0, 1, 2, 3], p=weight_list)
        if side_ix == 0:
            y = [-half_length, internal_RNG.uniform(-half_width, half_width)]
        elif side_ix == 1:
            y = [internal_RNG.uniform(-half_length, half_length), -half_width]
        elif side_ix == 2:
            y = [half_length, internal_RNG.uniform(-half_width, half_width)]
        elif side_ix == 3:
            y = [internal_RNG.uniform(-half_length, half_length), half_width]
        else:
            raise ValueError("Something went wrong during selection of side selection")
        # measurement y
        # rotate to match orientation
        y = rot(theta) @ y
        # offset based on location of ellipse center
        y += loc
        # save
        Y.append(y)
    # numpify Y
    Y = np.vstack(Y)

    # apply gaussian noise with cov. matrix R to all measurements
    if R is not None:
        Z = np.vstack([internal_RNG.multivariate_normal(y, R) for y in Y])
    else:
        Z = Y
    return Y, Z

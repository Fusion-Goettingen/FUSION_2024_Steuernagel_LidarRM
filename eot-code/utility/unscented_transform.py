"""
Implements functions related to the unscented transform
"""
import numpy as np
from scipy.linalg import sqrtm


def unscented_transform(mu, C, f, kappa=2., alpha=1., beta=2.):
    """
    Implements the unscented transform to estimate covariance after a (nonlinear) transformation.

    Automatically calculates sigma points and weights according to the scaled unscented transform

    For a comprehensive overview, check out:
    http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam06-ukf-4.pdf

    Parameters
    ----------
    mu : array_like
        Mean of the distribution passed to the unscented transform
    C : array_like
        Covariance matrix of the distribution passed to the unscented transform
    f : Callable[[array_like], array_like]
        Function that, given a point similar to m, returns a transformed version of this point.
    kappa: float
        kappa parameter of scaled UT, with kappa >= 0
    alpha: float
        alpha parameter of scaled UT, with 0 < alpha <= 1
    beta: float
        beta parameter of scaled UT. For Gaussians, should be 2.

    Returns
    -------
    array_like, array_like, array_like
        (mu_ut, C_ut, sigma_pts): Transformed mean, transformed covariance matrix, sigma points used
        The transformed sigma points are not returned by this function.
    """
    n = len(mu)
    
    # initial assertions on correctness
    assert 0 < alpha <= 1, "alpha must be in (0, 1] (is {})".format(alpha)
    assert kappa >= 0, "kappa must be >=0 (is {})".format(kappa)
    assert C.shape == (
        n, n
    ), "C must be of shape NxN with N the dim. of the mean (is {})".format(
        C.shape)
    
    # calculate internal parameter lambda
    lam = alpha**2 * (n + kappa) - n
    
    # calculate sigma points
    sigma_pts = [mu]
    for i in range(n):
        next_vec = sqrtm(((n + lam) * C))[:, i]
        sigma_pts.append(mu + next_vec)
        sigma_pts.append(mu - next_vec)
    sigma_pts = np.vstack(sigma_pts)

    # calculate weights
    w_m = [lam / (n + lam)]
    w_c = [w_m[0] + (1 - alpha**2 + beta)]
    for i in range(1, 2 * n + 1):
        w_m.append(1 / (2 * (n + lam)))
        w_c.append(1 / (2 * (n + lam)))

    # estimate mean:
    mu_dash = np.array([0, 0])
    for i in range(2 * n + 1):
        mu_dash = mu_dash + w_m[i] * f(sigma_pts[i])

    # estimate cov.:
    C_dash = np.zeros((n, n))
    for i in range(2 * n + 1):
        centered_transformed_point = (f(sigma_pts[i]) - mu_dash).reshape(
            (n, 1))
        C_dash = C_dash + w_c[
            i] * centered_transformed_point @ centered_transformed_point.T

    # return results
    return mu_dash, C_dash, sigma_pts

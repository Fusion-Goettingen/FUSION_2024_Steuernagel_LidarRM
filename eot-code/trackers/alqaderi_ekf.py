"""
Implments an EKF-based tracker proposed by Alqaderi et al.:

H. Alqaderi, F. Govaers, and R. Schulz,
“Spacial elliptical model for extended target tracking using laser measurements,”
in 2019 Sensor Data Fusion: Trends, Solutions, Applications (SDF), Oct. 2019, pp. 1–6.
"""
import numpy as np
from numpy import cos, sin
from numpy.linalg import inv
from scipy.stats import multivariate_normal as mvn

from trackers.abstract_tracking import AbtractEllipticalTracker
from utility.unscented_transform import unscented_transform
from utility.utils import cart2pol_pt, pol2cart_pt, rot

IX_LOC_X = 0
IX_LOC_Y = 1
IX_VEL_X = 2
IX_VEL_Y = 3
IX_L1 = 4
IX_L2 = 5


class AlqaderiEKF(AbtractEllipticalTracker):
    use_unscented_transform = True

    def __init__(self,
                 P_init,
                 R,
                 F,
                 Q,
                 n_gmm_components,
                 mode="permutation"):
        """
        Implements the tracker

        :param P_init: Initial cov of state
        :param R: Meas Noise
        :param F: State trans. matrix
        :param Q: Process noise cov
        :param n_gmm_components: Number of components to use across contour
        :param mode: "sequential" or "permutation". In "sequential" mode, measurements are directly processed in order.
        In "permutation" mode, they are processed after random shuffling of them. Note that in a lot of experiments
        carried out for this paper, the "sequential" mode causes filter divergence. If your measurements are randomly
        sampled from the contour, it should be slightly more efficient, but if a rotating lidar is used (or simulated),
        consider trying "permutation" mode.
        """
        self.P = P_init
        self.R = R
        self.F = F
        self.Q = Q
        self.x = None
        self._n_gmm_components = n_gmm_components
        self.mode = mode

    def _initialize(self, measurements):
        self.x = np.zeros((6,))
        self.x[[IX_LOC_X, IX_LOC_Y]] = np.mean(measurements, axis=0)

        # shape
        X = measurements - self.x[[IX_LOC_X, IX_LOC_Y]].reshape((-1, 2))
        X = X.T @ X
        X /= len(measurements)
        val, vec = np.linalg.eig(X)  # eigenvalue decomposition
        alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
        alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
        p = [alpha, *np.sqrt(val)]
        self.x[IX_L1] = p[1]
        self.x[IX_L2] = p[2]
        return self.get_state()

    def update(self, measurements: np.ndarray):
        if self.x is None:
            return self._initialize(measurements)

        if self.use_unscented_transform:
            _, R, _ = unscented_transform(mu=cart2pol_pt(np.mean(measurements, axis=0).reshape((-1,))),
                                          C=self.R,
                                          f=pol2cart_pt)
        else:
            R = self.R

        if self.mode == "sequential":
            for z in measurements:
                # run an individual measurement update
                self.x, self.P = self.single_meas_update(x_minus=self.x,
                                                         P_minus=self.P,
                                                         z=z,
                                                         R=R
                                                         )
        elif self.mode == "permutation":
            for z in measurements[np.random.permutation(range(len(measurements))), :]:
                # run an individual measurement update
                self.x, self.P = self.single_meas_update(x_minus=self.x,
                                                         P_minus=self.P,
                                                         z=z,
                                                         R=R
                                                         )
        else:
            raise NotImplementedError(f"{self.__class__.__name__} does not implement mode '{self.mode}'!")
        return self.get_state()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def get_state(self):
        state = np.zeros((7,))
        state[:4] = self.x[:4]
        state[4] = np.arctan2(self.x[IX_VEL_Y], self.x[IX_VEL_X])  # match heading to velocity
        state[5] = self.x[IX_L1]
        state[6] = self.x[IX_L2]
        return state

    def set_R(self, R):
        self.R = R

    def _g(self, x, theta):
        """Implements (5)"""
        alpha = np.arctan2(x[IX_VEL_Y], x[IX_VEL_X])
        C = np.array(x[[IX_LOC_X, IX_LOC_Y]])
        y = rot(alpha) @ np.diag(x[[IX_L1, IX_L2]]) @ np.array([cos(theta), sin(theta)]).reshape((2, 1))
        y = C + y.reshape((2,))
        return y

    def single_meas_update(self, x_minus, P_minus, z, R):
        """
        Run the update for a given point measurement z

        :param x_minus: Prior state
        :param P_minus: Prior covariance
        :param z: Measurement
        :param R: Measurement noise covariance
        :return: (x, P) the updated state and covariance
        """
        # prepare variables
        n = self._n_gmm_components
        theta = np.linspace(0, 2 * np.pi, num=n)
        v, W, S, x, P, weights = [], [], [], [], [], []

        # match heading to velocity
        alpha = np.arctan2(x_minus[IX_VEL_Y], x_minus[IX_VEL_X])
        for j in range(n):
            # compute Jacobian (13)
            J = np.array([
                [1, 0, 0, 0, cos(alpha) * cos(theta[j]), -sin(alpha) * sin(theta[j])],
                [0, 1, 0, 0, sin(alpha) * cos(theta[j]), cos(alpha) * sin(theta[j])],
            ])

            # prepare update helper variables (following (20))
            g = self._g(x_minus, theta[j])
            v_j = z - g
            S_j = J @ P_minus @ J.T + R
            try:
                W_j = P_minus @ J.T @ inv(S_j)
            except np.linalg.LinAlgError:
                # numerical instability, can't inv(S_j)
                continue

            # update component (19) and (20)
            x_j = x_minus + W_j @ v_j
            P_j = P_minus - W_j @ S_j @ W_j.T

            # Ensure semi-axis are non-negative  [not discussed in paper]
            eps = 0.05
            x_j[-2:][x_j[-2:] < eps] = eps

            # save
            v.append(v_j)
            W.append(W_j)
            S.append(S_j)
            x.append(x_j)
            P.append(P_j)

            # compute and save weight
            try:
                weights.append(mvn.pdf(z, mean=g, cov=S_j))
            except ValueError:
                # use small identity instead, numerical instability caused problems
                weights.append(mvn.pdf(z, mean=g, cov=np.eye(2) * 1e-3))
            except np.linalg.LinAlgError:
                # use small identity instead, numerical instability caused problems
                weights.append(mvn.pdf(z, mean=g, cov=np.eye(2) * 1e-3))

        # weights need to be normalized
        weights = np.array(weights)
        weights /= weights.sum()

        # weights sometimes become all zero? - potential indicator of filter divergence
        if np.isnan(weights).any():
            weights = np.full(shape=weights.shape,
                              fill_value=1 / len(weights))
        # moment matching of Gaussian Mixture to single Gaussian (17) (18)
        x_plus = np.average(x, axis=0, weights=weights)
        P_plus = np.sum([
            weights[j] * (P[j] + (x[j] - x_plus).reshape((-1, 1)) @ (x[j] - x_plus).reshape((-1, 1)).T)
            for j in range(len(weights))
        ], axis=0)
        return x_plus, P_plus

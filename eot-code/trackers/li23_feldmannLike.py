"""
Implements an adapted version of the modified random matrix tracker for 2D LiDAR, proposed by Li et al. (2023)

Original article:
P. Li, C. Chen, C. You, and J. Qiu,
“Modified extended object tracker for 2D lidar data using random matrix model,”
Sci Rep, vol. 13, no. 1, Art. no. 1, Mar. 2023, doi: 10.1038/s41598-023-32236-w.


Adaption to using the "standard" Feldmann Formulas

c.f.:
A Tutorial on Multiple Extended Object Tracking
K. Granström and M. Baum, 2022
https://www.techrxiv.org/articles/preprint/ATutorialonMultipleExtendedObjectTracking/19115858/1

Tracking of Extended Objects and Group Targets Using Random Matrices
M. Feldmann, D. Fränken, W. Koch, 2011
https://ieeexplore.ieee.org/document/5672614
"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

from trackers.abstract_tracking import AbtractEllipticalTracker
from utility.unscented_transform import unscented_transform
from utility.utils import cart2pol_pt, pol2cart_pt


class LiRMLikeFeldmann(AbtractEllipticalTracker):
    """
    Implements a tracker based on the Random Matrix (RM) model
    """
    use_unscented_transform = True

    def __init__(self,
                 P,
                 v,
                 R,
                 lidar_kwargs=None,
                 H=None,
                 Q=None,
                 time_step_length=1,
                 nu=5,
                 uniform_weights=True,
                 tau=10,
                 z_scale=2 / 3):
        """

        :param m: Initial kinematic state
        :param P: Initial kinematic state uncertainty
        :param v: Extent uncertainty
        :param V: Initial extent estimate
        :param R: Measurement noise
        :param H: Measurement model
        :param Q: Process noise
        :param time_step_length: time between discrete time steps
        :param tau: hyperparameter that determines decay of v. use large values if you know the object shape barely
        changes over time
        :param z_scale: Scaling factor. Choose 0.25 for approximating a uniform distribution on an elliptical target
        """
        self.m = None
        self.P = P
        self.v = v
        self.V = None

        self.tau = tau
        self.lidar_kwargs = lidar_kwargs
        self.nu = nu
        self.uniform_weights = uniform_weights

        # Two Dimensional:
        self._time_step_length = time_step_length
        self._d = 2
        self.z_scale = z_scale

        self.H = np.array(H) if H is not None else np.hstack([np.eye(2), np.zeros((2, len(P) - 2))])
        self.R = np.array(R)
        self.Q = Q if Q is not None else np.zeros(self.P.shape)

        assert self.Q.shape == self.P.shape, f"Q and P shape don't align ({self.Q.shape} vs {self.P.shape})"

        # DEBUG
        self.last_expected = None

    def predict(self):
        assert self.m is not None and self.V is not None, "m or V is None - did you predict before first update?"
        self.m, self.P, self.v, self.V = self.predict_rm(self.m, self.P, self.v, self.V)
        return self.get_state()

    def update(self, Z):
        if self.m is None or self.V is None:
            m = np.average([Z[0], Z[-1]], axis=0)
            self.m = np.array([*m, 0, 0])
            # get cov mat
            X = Z - m.reshape((-1, 2))
            X = X.T @ X

            X /= len(Z)
            self.V = 2 * (self.v - self._d - 1) * X * self.z_scale

            return self.get_state()
        else:
            self.m, self.P, self.v, self.V = self.update_rm(Z, self.m, self.P, self.v, self.V)
            return self.get_state()

    def set_R(self, R):
        self.R = R

    def get_state(self):
        x, y = self.m.reshape((-1,))[:2]
        orientation, length, width = self._matrix_to_params_rm(self.V / (self.v - self._d - 1))
        # convert semi-axis length to axis length
        length, width = length, width
        velo_x, velo_y = self.m.reshape((-1,))[2:4]
        state = np.array([x, y, velo_x, velo_y, orientation, length, width]).astype(float)
        return state

    @staticmethod
    def _matrix_to_params_rm(X):
        """Convert shape matrix X to parameter form [alpha, l1, l2] with semi-axis length"""
        assert X.shape == (2, 2), "X is not a 2x2 matrix"
        val, vec = np.linalg.eig(X)  # eigenvalue decomposition
        alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
        alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
        p = [alpha, *np.sqrt(val)]
        return np.array(p)

    def _compute_segment_midpoints(self, measurements):
        r = np.arctan2(measurements[:, 1], measurements[:, 0]) % (2 * np.pi)
        segments = np.linspace(0, 2 * np.pi, self.lidar_kwargs["horizontal_resolution"])
        digitized = np.digitize(r, segments)
        bin_means = np.array([measurements[digitized == i, :].mean(axis=0) for i in range(1, len(segments))])
        filled_segments, _ = np.where(np.isfinite(bin_means))
        first_detected_segment, last_detected_segment = np.min(filled_segments), np.max(filled_segments)
        relevant_segments = bin_means[first_detected_segment:last_detected_segment + 1]
        midpoints = []
        for i in range(int(len(relevant_segments) / 2)):  # int casting entirely avoids middle segment
            # match 1st with last, 2nd with 2nd to last
            left_segment_mean = relevant_segments[i]
            # note: python backwards indexing has -1 as last -> add +1
            right_segment_mean = relevant_segments[- (i + 1)]
            if np.sum(np.isnan([left_segment_mean, right_segment_mean])):
                # either segment didn't contain points -> digitize returned nan
                continue
            midpoints.append(0.5 * (left_segment_mean + right_segment_mean))

        return np.array(midpoints)

    def _compute_weights(self, z):
        n_k = len(z)
        if self.uniform_weights or z.shape[0] == 1:
            return np.full(shape=(n_k,), fill_value=1)
        else:
            omega = np.zeros(n_k)
            d = np.linalg.norm(z, axis=1)
            r = np.arctan2(z[:, 1], z[:, 0])
            distances_between_measurements = cdist(z, z)
            distances_between_measurements[np.diag_indices_from(distances_between_measurements)] = float("inf")
            for i in range(n_k):
                # measurement closest to z_i
                j = np.argmin(distances_between_measurements[i, :])
                d_hat = min(d[i], d[j])
                diff = (z[i] - z[j]).reshape((2, 1))
                omega[i] = (diff.T @ diff) / (d_hat ** 2 * (2 - 2 * np.cos(np.abs(r[i] - r[j]))))
            tau = n_k / np.sum(omega)
            omega *= tau
            return omega

    def get_tangent_orthogonal_line_coefficients(self, measurements):
        xL, yL = 0, 0  # sensor position, here fix at [0, 0]
        x1, y1 = measurements[0]
        xnk, ynk = measurements[-1]
        A1 = - (x1 - xL) / (y1 - yL)
        B1 = y1 - A1 * x1
        A2 = - (xnk - xL) / (ynk - yL)
        B2 = ynk - A2 * xnk
        return A1, B1, A2, B2

    def compute_z_bar(self, measurements):
        z = measurements
        n_k = len(z)

        omega = self._compute_weights(measurements)
        # Z_bar is the set of matched segment means
        Z_bar = self._compute_segment_midpoints(measurements)
        n_bar = len(Z_bar)
        if n_bar <= self.nu:
            z_bar = np.average(z, weights=omega, axis=0)
        else:
            A1, B1, A2, B2 = self.get_tangent_orthogonal_line_coefficients(measurements)

            # compute C
            C = np.ones((n_bar, 2))  # start with 1s everywhere
            C[:, 1] = Z_bar[:, 0]
            D = Z_bar[:, 1].reshape((n_bar, 1))
            # extract least squares solution for A/B_3
            # note that the order must be inverted compared to what is given in the paper
            B3, A3 = np.linalg.inv(C.T @ C) @ C.T @ D

            mu_1 = np.array([
                (B3 - B1) / (A1 - A3),
                A1 * ((B3 - B1) / (A1 - A3)) + B1
            ])
            mu_2 = np.array([
                (B3 - B2) / (A2 - A3),
                A2 * ((B3 - B2) / (A2 - A3)) + B2
            ])
            z_bar = np.mean([mu_1, mu_2], axis=0).reshape((-1,))
        return z_bar

    def update_rm(self, measurements, m_minus, P_minus, v_minus, V_minus):
        """
        Update function for the RM model, based on a batch of measurements

        :param measurements: Measurement batch
        :param m_minus: prior kinematic state
        :param P_minus: prior kinematic covariance
        :param v_minus: prior extent uncertainty
        :param V_minus: prior extent estimate
        :return: m_plus, P_plus, v_plus, V_plus - updated posterior estimates
        """
        # pre-process
        assert v_minus > 2 * self._d + 2  # if v_minus is too small, the X_hat calculation will cause problems
        m_minus = np.reshape(m_minus, (len(m_minus), -1))

        # Begin update
        # z_bar = np.average(W, axis=0).reshape((2, 1))  # standard random matrix
        z_bar = self.compute_z_bar(measurements).reshape((2, 1))
        if self.use_unscented_transform:
            _, R, _ = unscented_transform(mu=cart2pol_pt(z_bar.reshape((-1,))), C=self.R, f=pol2cart_pt)
        else:
            R = self.R

        m = m_minus[:2].reshape((2, 1))
        K_temp = P_minus[:2, :2] @ np.linalg.inv(P_minus[:2, :2] + R / 2)
        z_bar_predicted = m + K_temp @ (z_bar - m_minus[:2])

        # matrix-based calculation of Z: (zi-z_bar)(zi-z_bar)^T
        Z = measurements - z_bar.reshape((-1, 2))
        Z = Z.T @ Z

        e = z_bar - self.H @ m_minus
        e = np.reshape(e, (-1, 1))

        X_hat = V_minus * (v_minus - 2 * self._d - 2) ** (-1)
        Y = self.z_scale * X_hat + R
        S = self.H @ P_minus @ self.H.T + Y / len(measurements)
        S_inv = np.linalg.inv(S)
        K = P_minus @ self.H.T @ S_inv

        # using astype(float) to ensure that numerical issues don't cause complex-valued results
        X_2 = np.array(sqrtm(X_hat)).astype(float)
        S_i2 = np.array(sqrtm(S_inv)).astype(float)
        Y_i2 = np.array(sqrtm(np.linalg.inv(Y))).astype(float)

        N_hat = X_2 @ S_i2 @ e @ e.T @ S_i2.T @ X_2.T
        Z_hat = X_2 @ Y_i2 @ Z @ Y_i2.T @ X_2.T

        m_plus = m_minus.reshape((-1, 1)) + K @ e
        P_plus = P_minus - K @ S @ K.T
        v_plus = v_minus + len(measurements)
        V_plus = V_minus + N_hat + Z_hat

        return m_plus, P_plus, v_plus, V_plus

    def get_F(self, T):
        """
        Helper function returning a constant velocity motion model matrix F given T
        :param T: time step length
        :return: F as ndarray
        """
        F = np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F

    def predict_rm(self, m_minus, P_minus, v_minus, V_minus):
        """
        Predict function for the RM model
        :param m_minus: prior kinematic state
        :param P_minus: prior kinematic covariance
        :param v_minus: prior extent uncertainty
        :param V_minus: prior extent estimate
        :return: m_plus, P_plus, v_plus, V_plus - predicted estimates
        """
        # parameters:
        T = self._time_step_length

        F = self.get_F(T)

        # kinematics
        m_plus = F @ m_minus
        P_plus = F @ P_minus @ F.T + self.Q

        # shape:
        # decay v_minus by e^(-T/tau)
        # to prevent v_plus from being too small: only decay a portion greater than 2*d-2
        v_plus = np.exp(-T / self.tau) * (v_minus - 2 * self._d - 2)
        v_plus += 2 * self._d + 2

        V_plus = ((v_plus - self._d - 1) / (v_minus - self._d - 1)) * V_minus
        return m_plus, P_plus, v_plus, V_plus

    def get_state_and_cov(self):
        raise NotImplementedError

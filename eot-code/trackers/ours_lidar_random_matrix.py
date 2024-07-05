import numpy as np
from scipy.linalg import sqrtm

from trackers.abstract_tracking import AbtractEllipticalTracker
from utility.unscented_transform import unscented_transform
from utility.utils import cart2pol_pt, pol2cart_pt
from utility.data_generation import get_rect_contour_measurements
from utility.tracking_helpers import get_max_angular_distance_pair


class FeldmannAdaptedToLidar(AbtractEllipticalTracker):
    """
    Implements a tracker based on the Random Matrix (RM) model adapted to 2D Lidar data
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
        :param z_scale: Scaling factor
        """
        self.m = None
        self.P = P
        self.v = v
        self.V = None

        self.tau = tau
        self.lidar_kwargs = lidar_kwargs

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
            m = self.compute_z_bar(Z)
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
        p = [alpha, *np.sqrt(np.abs(val))]
        return np.array(p)

    def compute_expected_measurement_offset(self):
        assert self.lidar_kwargs is not None
        x = self.get_state()
        Z_expected = get_rect_contour_measurements(x[:2],
                                                   x[4:],
                                                   horizontal_resolution=self.lidar_kwargs["horizontal_resolution"] * 4,
                                                   R=self.lidar_kwargs["R"],
                                                   polar_noise=True)
        return x[:2] - np.mean(Z_expected, axis=0), Z_expected

    def compute_z_bar(self, W, mode=None):
        # if the object is near the x-axis, the naive-intersect assumption does not hold
        # instead, we find the pair of points with the maximum angular distance, and intersect those
        # due to measurement noise, this is suboptimal if using the order of points is possible
        # (note: this is due to wrapping around 2pi and the rotating nature of the lidar)

        # detect whether target is intersecting x-Axis on the positive side
        is_on_x_axis = np.min(np.abs(W[:, 1])) < 0.05 and np.min(W[:, 0]) > 0
        if is_on_x_axis:
            # find max-angular dist. pair
            ix0, ix1 = get_max_angular_distance_pair(W)
            z_bar = np.average([W[ix0], W[ix1]], axis=0)
        else:
            # use point order
            z_bar = np.average([W[0], W[-1]], axis=0)
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

        # matrix-based calculation of Z: (zi-z_bar)(zi-z_bar)^T
        Z = measurements - z_bar.reshape((-1, 2))
        Z = Z.T @ Z

        est_semi_axis = self._matrix_to_params_rm(Z / len(measurements))[1:]

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

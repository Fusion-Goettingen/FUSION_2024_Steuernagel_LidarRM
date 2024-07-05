"""
Implements the MEM-EKF* extended via a Gaussian mixture measurement source distribution.

The original MEM-EKF* is
presented in
Yang, Shishan, and Marcus Baum. "Tracking the orientation and axes lengths of an elliptical extended object." IEEE
Transactions on Signal Processing 67.18 (2019): 4720-4729.

The so called GM-MEM-EKF* using a Gaussian mixture to model the distribution of measurement sources is presented in
Thormann, Kolja, Shishan Yang, and Marcus Baum. "Kalman Filter Based Extended Object Tracking with a Gaussian Mixture
Spatial Distribution Model." 2021 IEEE Intelligent Vehicles Symposium Workshops (IV Workshops). IEEE, 2021.
"""

import numpy as np
from numpy.linalg import slogdet
from scipy.special import logsumexp

from trackers.abstract_tracking import AbtractEllipticalTracker
from utility.unscented_transform import unscented_transform
from utility.utils import cart2pol_pt, pol2cart_pt
from utility.utils import rot

MIN_INIT_AXIS_LENGTH = 1.0
MIN_AXIS_LENGTH = 0.1


class MEMEKF(AbtractEllipticalTracker):
    def __init__(self,
                 cov_kin,
                 cov_shape,
                 measurement_noise_cov_polar,
                 kin_process_cov,
                 shape_process_cov,
                 sensor_position=np.zeros(2),
                 z_scale=1 / 3,
                 permute_measurements=True,
                 ):
        """
        Create a new GM-MEM-EKF* instance
        :param state_kin: 4D kinematic state consisting of center and Cartesian velocity [m1, m2, v1, v2]
        :param cov_kin: 4x4 covariance of the kinematic state
        :param state_shape: 3D shape state consisting of orientation and semi-axis lengths [theta, l, w]
        :param cov_shape: 3x3 covariance matrix of the shape state
        :param measurement_noise_cov_polar: 2x2 measurement noise covariance
        :param kin_process_cov: 4x4 process noise covariance for the kinematic state assuming delta_t=1
        :param shape_process_cov: 3x3 process noise covariance for the shape state assuming delta_t=1
        :param sensor_position: 2D position of the sensor
        """
        # parameters
        self._h_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        self._measurement_noise_cov_polar = measurement_noise_cov_polar
        self._measurement_noise_cov = measurement_noise_cov_polar

        # transition noise and matrices, assuming delta_t=1
        self._kin_tran_matrix = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._kin_process_cov = kin_process_cov
        self._shape_tran_matrix = np.eye(3)
        self._shape_process_cov = shape_process_cov

        self._multiplicative_noise_cov = np.diag([1.0 / 3.0, 1.0 / 3.0])

        self._state_kin = None
        self._cov_kin = cov_kin
        self._state_shape = None
        self._cov_shape = cov_shape

        self._gm_h_means = np.array([
            [0.99, 0.0],
            [0.0, 0.99],
            [-0.99, 0.0],
            [0.0, -0.99],
        ])
        # approximate uniform distribution across entire side and with a width corresponding to 2*(1.0-h_mean)
        self._gm_h_var = np.array([
            np.diag([1.0 / 30000.0, 1.0 / 3.0]),
            np.diag([1.0 / 3.0, 1.0 / 30000.0]),
            np.diag([1.0 / 30000.0, 1.0 / 3.0]),
            np.diag([1.0 / 3.0, 1.0 / 30000.0]),
        ])
        self._gm_h_weight = np.ones(4) * 0.25

        self._sensor_position = sensor_position

        self._z_scale = z_scale

        self._use_unscented_transform = True

        self._use_mmgw_estimate = False
        self._rng = np.random.default_rng(512)
        self._permute_measurements = permute_measurements

    def update(self, measurements: np.ndarray):
        """
        Perform an update given measurements for the current time step.

        If no prior update was done, this function is used to initialize the tracker estimates.

        :param measurements: N measurements as Nx2 numpy ndarray
        :return: 7D array [x, y, vx, vy, theta, semi-axis length, semi-axis width]
        """
        if self._state_kin is None or self._state_shape is None:
            m = np.average(measurements, axis=0)
            self._state_kin = np.array([*m, 0, 0])
            # get cov mat
            X = measurements - m.reshape((-1, 2))
            X = X.T @ X / len(measurements)
            eigvals, eigvecs = np.linalg.eig(X / self._z_scale)
            self._state_shape = np.array([
                np.arctan2(eigvecs[1, 0], eigvecs[0, 0]),
                np.maximum(np.sqrt(eigvals[0]), MIN_INIT_AXIS_LENGTH),
                np.maximum(np.sqrt(eigvals[1]), MIN_INIT_AXIS_LENGTH),
            ])
            return self.get_state()
        else:
            if self._permute_measurements:
                measurements = measurements[self._rng.permutation(range(len(measurements))), :]
            self.update_memekf(measurements)
            return self.get_state()

    def update_memekf(self, measurements):
        if len(measurements) == 0:
            return self.get_state()

        if self._use_unscented_transform:
            _, self._measurement_noise_cov, _ \
                = unscented_transform(mu=cart2pol_pt(np.average(measurements, axis=0).reshape((-1,))),
                                      C=self._measurement_noise_cov_polar,
                                      f=pol2cart_pt)

        sensor_location_local = self._sensor_position - self._state_kin[:2]
        sensor_location_local = rot(self._state_shape[0]).T @ sensor_location_local
        sensor_location_local = np.diag(1.0 / self._state_shape[1:3]) @ sensor_location_local
        sensor_direction = np.arctan2(sensor_location_local[1], sensor_location_local[0])

        for i in range(len(self._gm_h_means)):
            h_dir = np.arctan2(self._gm_h_means[i, 1], self._gm_h_means[i, 0])
            dir_diff = abs((((sensor_direction - h_dir) + np.pi) % (2.0 * np.pi)) - np.pi)
            if dir_diff <= np.pi / 8:
                self._gm_h_weight[i] = 1.0
            elif dir_diff <= 3 * np.pi / 8:
                self._gm_h_weight[i] = 0.8
            else:
                self._gm_h_weight[i] = 0.0
        self._gm_h_weight /= np.sum(self._gm_h_weight)

        for i, y in enumerate(measurements):
            # prepare Gaussian mixture posterior
            state_kins = np.zeros((len(self._gm_h_means), len(self._state_kin)))
            cov_kins = np.zeros((len(self._gm_h_means), len(self._state_kin), len(self._state_kin)))
            state_shapes = np.zeros((len(self._gm_h_means), len(self._state_shape)))
            cov_shapes = np.zeros((len(self._gm_h_means), len(self._state_shape), len(self._state_shape)))
            result_weights = np.log(self._gm_h_weight)

            # iterate over mixture components
            for j in range(len(self._gm_h_means)):
                c1, c2, f_matrix, f_dash_matrix, m_matrix, s = self.get_aux_variables(comp_idx=j)

                # calculate moments for the kinematic state update
                y_dash = self._h_matrix @ self._state_kin + np.dot(s, self._gm_h_means[j])
                cov_ry = self._cov_kin @ self._h_matrix.T
                cov_y = self._h_matrix @ self._cov_kin @ self._h_matrix.T + c1 + c2 \
                        + self._measurement_noise_cov_polar[:2, :2]

                # update kinematic state
                y_dif = y[:2] - y_dash
                cov_y_inv = np.linalg.inv(cov_y)  # bias only for kin, not for shape
                gain = cov_ry @ cov_y_inv
                state_kins[j] = self._state_kin + gain @ y_dif
                cov_kins[j] = self._cov_kin - cov_ry @ cov_y_inv @ cov_ry.T
                cov_kins[j] = (cov_kins[j] + cov_kins[j].T) / 2.0  # enforces symmetry of the covariance

                # construct pseudo-measurement for the extent update
                pseudo_meas = f_matrix @ np.kron(y_dif, y_dif)
                # calculate moments for the extent update
                pseudo_meas_xpctn = f_matrix @ cov_y.flatten()
                cov_extent_pseudo_meas = self._cov_shape @ m_matrix.T
                cov_pseudo_meas = f_matrix @ np.kron(cov_y, cov_y) @ (f_matrix + f_dash_matrix).T

                # update extent
                inv_cov_pseudo = np.linalg.inv(cov_pseudo_meas)
                gain_p = cov_extent_pseudo_meas @ inv_cov_pseudo
                innov_p = pseudo_meas - pseudo_meas_xpctn

                state_shapes[j] = self._state_shape + gain_p @ innov_p

                cov_shapes[j] = self._cov_shape - cov_extent_pseudo_meas @ inv_cov_pseudo @ cov_extent_pseudo_meas.T
                cov_shapes[j] = (cov_shapes[j] + cov_shapes[j].T) / 2.0  # enforces symmetry of the covariance

                l_lik = -len(y_dif) * (np.log(2 * np.pi) + 0.5 * slogdet(cov_y)[1])
                l_lik -= 0.5 * (y_dif @ np.linalg.inv(cov_y) @ y_dif)
                result_weights[j] += l_lik

            result_weights -= logsumexp(result_weights)
            result_weights = np.exp(result_weights)

            # moment matching (based on KL minimization, e.g., Granström and Orguner, "On the Reduction of Gaussian
            # inverse Wishart Mixtures")
            self._state_kin = np.sum(result_weights[:, None] * state_kins, axis=0)
            self._cov_kin = np.sum(result_weights[:, None, None]
                                   * (cov_kins + np.einsum('xa, xb -> xab', state_kins - self._state_kin,
                                                           state_kins - self._state_kin)), axis=0)
            self._state_shape = np.sum(result_weights[:, None] * state_shapes, axis=0)
            self._cov_shape = np.sum(result_weights[:, None, None]
                                     * (cov_shapes + np.einsum('xa, xb -> xab', state_shapes - self._state_shape,
                                                               state_shapes - self._state_shape)), axis=0)

            self._state_shape = np.array([
                ((self._state_shape[0] + np.pi) % (2 * np.pi)) - np.pi,
                np.maximum(self._state_shape[1], MIN_AXIS_LENGTH),
                np.maximum(self._state_shape[2], MIN_AXIS_LENGTH),
            ])

    def get_aux_variables(self, comp_idx, shape_prior=None, shape_cov_prior=None):
        """
        Create variables for correct step.
        :return:        Parts of noise covariance, f matrices, and M matrix
        """
        h_t = self._gm_h_means[comp_idx],
        h_t_var = self._gm_h_var[comp_idx]

        shape_prior = self._state_shape if shape_prior is None else shape_prior
        shape_cov_prior = self._cov_shape if shape_cov_prior is None else shape_cov_prior

        alpha, l1, l2 = shape_prior
        sin = np.sin(alpha)
        cos = np.cos(alpha)

        s = np.array([[cos, -sin], [sin, cos]]) @ np.diag([l1, l2])
        s1 = np.array([s[0]])
        s2 = np.array([s[1]])

        j1 = np.array([[-l1 * sin, cos, 0], [-l2 * cos, 0, -sin]])
        j2 = np.array([[l1 * cos, sin, 0], [-l2 * sin, 0, cos]])

        c1 = s @ h_t_var @ s.T
        e11 = np.trace(shape_cov_prior @ j1.T @ (h_t_var + np.outer(h_t, h_t)) @ j1)
        e12 = np.trace(shape_cov_prior @ j2.T @ (h_t_var + np.outer(h_t, h_t)) @ j1)
        e22 = np.trace(shape_cov_prior @ j2.T @ (h_t_var + np.outer(h_t, h_t)) @ j2)
        c2 = np.array([[e11, e12], [e12, e22]])

        f_matrix = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 1, 0, 0]])
        f_dash_matrix = np.array([[1, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 1, 0]])

        m_matrix = np.vstack([2 * s1 @ h_t_var @ j1,
                              2 * s2 @ h_t_var @ j2,
                              s1 @ h_t_var @ j2
                              + s2 @ h_t_var @ j1])

        return c1, c2, f_matrix, f_dash_matrix, m_matrix, s

    def predict(self):
        """
        Perform a predict step for delta-t = 1
        :return: 7D array [x, y, vx, vy, theta, semi-axis length, semi-axis width]
        """
        self._state_kin = self._kin_tran_matrix @ self._state_kin
        self._cov_kin = self._kin_tran_matrix @ self._cov_kin @ self._kin_tran_matrix.T + self._kin_process_cov

        self._state_shape = self._shape_tran_matrix @ self._state_shape
        self._cov_shape = self._shape_tran_matrix @ self._cov_shape @ self._shape_tran_matrix.T + self._shape_process_cov

        return self.get_state()

    def get_state(self):
        """
        Return the current state as 7D array [x, y, vx, vy, theta, semi-axis length, semi-axis width]
        :return: 7D array [x, y, vx, vy, theta, semi-axis length, semi-axis width]
        """
        if self._use_mmgw_estimate:
            samples = self._rng.multivariate_normal(self._state_shape, self._cov_shape, 1000)
            sample_mats = np.array([rot(samples[i, 0]) @ np.diag(samples[i, 1:3]) @ rot(samples[i, 0]).T
                                    for i in range(1000)])
            shape_mat = np.mean(sample_mats, axis=0)
            ellipse_axis, v = np.linalg.eig(shape_mat)
            ax_l = ellipse_axis[0]
            ax_w = ellipse_axis[1]
            al = np.arctan2(v[1, 0], v[0, 0])

            return np.hstack([self._state_kin, [al, ax_l, ax_w]])
        else:
            return np.hstack([self._state_kin, self._state_shape])

    def set_R(self, R):
        """
        Update the measurement noise covariance to a new value
        :param R: new measurement noise covariance
        """
        self._measurement_noise_cov_polar = R

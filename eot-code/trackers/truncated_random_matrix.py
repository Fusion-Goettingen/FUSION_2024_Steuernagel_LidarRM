"""
Implements the truncated random matrix tracker. Please refer to
Xia, Yuxuan, et al. "Learning-based extended object tracking using hierarchical truncation measurement model with
automotive radar." IEEE Journal of Selected Topics in Signal Processing 15.4 (2021): 1013-1029.
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm as norm_pdf

from trackers.feldmann_random_matrix import FeldmannRMTracker
from utility.unscented_transform import unscented_transform
from utility.utils import cart2pol_pt, pol2cart_pt


class TruncatedRMTracker(FeldmannRMTracker):
    def __init__(self,
                 P,
                 v,
                 R,
                 H=None,
                 Q=None,
                 time_step_length=1,
                 tau=10,
                 z_scale=1 / 3,
                 trunc_iter=3,
                 sensor_position=np.zeros(2),
                 trunc_value=0.99):
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
        :param trunc_iter: number of iterations for conducting the update
        :param sensor_position: 2D array, position of the sensor in global coordinates
        :param trunc_value: percentage of visible side's volume that is self occluded
        """
        super().__init__(P, v, R, H, Q, time_step_length, tau, z_scale)

        self._trunc_iter = trunc_iter
        self._sensor_position = sensor_position
        self._trunc_value = trunc_value

    def update_rm(self, W, m_minus, P_minus, v_minus, V_minus):
        """
        Modified version of the random matrix update, using a truncated density

        :param W: Measurement batch
        :param m_minus: prior kinematic state
        :param P_minus: prior kinematic covariance
        :param v_minus: prior extent uncertainty
        :param V_minus: prior extent estimate
        :return: m_plus, P_plus, v_plus, V_plus - updated posterior estimates
        """
        # pre-process
        assert v_minus > 2 * self._d + 2  # if v_minus is too small, the X_hat calculation will cause problems
        m_minus = np.reshape(m_minus, (len(m_minus), -1))
        z_bar = np.average(W, axis=0).reshape((2, 1))
        num_m = len(W)
        if self.use_unscented_transform:
            _, R, _ = unscented_transform(mu=cart2pol_pt(z_bar.reshape((-1,))), C=self.R, f=pol2cart_pt)
        else:
            R = self.R

        # optimization
        m_plus = m_minus.copy()
        P_plus = P_minus.copy()
        v_plus = v_minus.copy()
        V_plus = V_minus.copy()
        for i in range(self._trunc_iter):
            # get shape parameters
            shape_eig, shape_rot = np.linalg.eig(1.0 / 3.0 * V_plus / (v_plus - 2 * self._d - 2))
            shape_eig[shape_eig < 1e-10] = 1e-10
            shape_rot = shape_rot.astype(float)
            boundary_unscaled = self.__get_boundary(m_plus, shape_rot)

            num_m_p = self.__get_number_of_pseudo_measurements(num_m, boundary_unscaled)
            h_t_hat = self.__get_mean_of_truncated_part(boundary_unscaled, shape_eig)
            h_cov = self.__get_covariance_of_truncated_part(boundary_unscaled, shape_eig)

            # pseudo measurement mean scaled by required number of pseudo measurements
            sum_m_p = num_m_p * (shape_rot @ h_t_hat * np.sqrt(self.z_scale) + self.H @ m_plus)

            # pseudo measurement spread
            sum_p_spread = num_m_p * (sum_m_p @ sum_m_p.T / num_m_p ** 2
                                      + self.z_scale * shape_rot @ h_cov @ shape_rot.T
                                      + R[:2, :2])

            # total mean and spread of the measurements combined with the pseudo-measurements
            mean_m_total = (np.sum(W, axis=0).reshape((2, 1)) + sum_m_p) / (num_m + num_m_p)
            spread_m_total = np.einsum('xa, xb -> ab', W - mean_m_total.T,
                                       W - mean_m_total.T) \
                             + num_m_p * (mean_m_total @ mean_m_total.T) + sum_p_spread \
                             - (sum_m_p @ mean_m_total.T) - (mean_m_total @ sum_m_p.T)

            # Begin update
            e = mean_m_total - self.H @ m_minus
            e = np.reshape(e, (-1, 1))

            X_hat = V_minus * (v_minus - 2 * self._d - 2) ** (-1)
            Y = self.z_scale * X_hat + R
            S = self.H @ P_minus @ self.H.T + Y / (num_m + num_m_p)
            S_inv = np.linalg.inv(S)
            K = P_minus @ self.H.T @ S_inv

            # using astype(float) to ensure that numerical issues don't cause complex-valued results
            X_2 = np.array(sqrtm(X_hat)).astype(float)
            S_i2 = np.array(sqrtm(S_inv)).astype(float)
            Y_i2 = np.array(sqrtm(np.linalg.inv(Y))).astype(float)

            N_hat = X_2 @ S_i2 @ e @ e.T @ S_i2.T @ X_2.T
            Z_hat = X_2 @ Y_i2 @ spread_m_total @ Y_i2.T @ X_2.T

            m_plus = (m_minus.reshape((-1, 1)) + K @ e).astype(float)
            P_plus = P_minus - K @ S @ K.T
            v_plus = v_minus + num_m + num_m_p
            V_plus = V_minus + N_hat + Z_hat

        return m_plus, P_plus, v_plus, V_plus

    def __get_boundary(self, m_plus, shape_rot):
        """
        Prepare boundaries (a rectangle that is cut out of the Gaussian)
        :param m_plus: at least 2D array of current estimate of the detected object's center
        :param shape_rot: 2x2 rotation matrix based on the detected object's orientation
        :return: 4D unscaled boundaries of the truncation area (left, right, bottom, top)
        """
        sensor_dir = ((np.arctan2(self._sensor_position[1] - m_plus[1], self._sensor_position[0] - m_plus[0])
                       - np.arctan2(shape_rot[1, 0], shape_rot[0, 0])
                       + np.pi) % (2.0 * np.pi)) - np.pi

        boundary_unscaled = np.zeros(4)
        boundary_unscaled[0] = -1.0 * self._trunc_value if abs(sensor_dir) >= 5 * np.pi / 8 else -np.inf
        boundary_unscaled[1] = self._trunc_value if abs(sensor_dir) <= 3 * np.pi / 8 else np.inf
        boundary_unscaled[2] = -1.0 * self._trunc_value if abs(sensor_dir + 0.5 * np.pi) <= 3 * np.pi / 8 else -np.inf
        boundary_unscaled[3] = self._trunc_value if abs(sensor_dir - 0.5 * np.pi) <= 3 * np.pi / 8 else np.inf

        return boundary_unscaled

    def __get_number_of_pseudo_measurements(self, num_m, boundary_unscaled):
        """
        Provides the number of pseudo measurements necessary given the probability mass ratio and the number of actual
        measurements.
        :param num_m: number of measurements
        :param boundary_unscaled: 4D unscaled boundary of the truncation area centered on the origin (left, right, bottom, top)
        :return: number of pseudo measurements
        """
        # probability that a sample lies in the area left after truncation
        c_d = (1.0 - (norm_pdf.cdf(boundary_unscaled[1]) - norm_pdf.cdf(boundary_unscaled[0]))
               * (norm_pdf.cdf(boundary_unscaled[3]) - norm_pdf.cdf(boundary_unscaled[2])))
        # necessary number of pseudo-measurements given number of actual measurements and probability mass ratio
        num_m_p = num_m * (1.0 - c_d) / c_d

        return num_m_p

    def __get_mean_of_truncated_part(self, boundary_unscaled, shape_eig):
        """
        Calculate the mean of the truncated part, assuming the original Gaussian is axis aligned.
        :param boundary_unscaled: 4D unscaled boundary of the truncation area centered on the origin (left, right, bottom, top)
        :param shape_eig: 2D array containing eigenvalues of the original Gaussians covariance matrix
        :return: 2D mean of the truncated part
        """
        return np.array(
            [np.sqrt(shape_eig[0]) * (norm_pdf.pdf(boundary_unscaled[0]) - norm_pdf.pdf(boundary_unscaled[1]))
             / (norm_pdf.cdf(boundary_unscaled[1]) - norm_pdf.cdf(boundary_unscaled[0])),
             np.sqrt(shape_eig[1]) * (norm_pdf.pdf(boundary_unscaled[2]) - norm_pdf.pdf(boundary_unscaled[3]))
             / (norm_pdf.cdf(boundary_unscaled[3]) - norm_pdf.cdf(boundary_unscaled[2]))]).reshape((2, 1))

    def __get_covariance_of_truncated_part(self, boundary_unscaled, shape_eig):
        """
        Calculate the covariance of the truncated part, assuming the original Gaussian is axis aligned.
        :param boundary_unscaled: 4D unscaled boundary of the truncation area centered on the origin (left, right, bottom, top)
        :param shape_eig: 2D array containing eigenvalues of the original Gaussians covariance matrix
        :return: 2x2 diagonal covariance matrix of the truncated part
        """
        return np.diag([
            self.__get_variance_of_single_dimension_of_truncated_part(boundary_unscaled[:2], shape_eig[0]),
            self.__get_variance_of_single_dimension_of_truncated_part(boundary_unscaled[2:4], shape_eig[1])
        ])

    def __get_variance_of_single_dimension_of_truncated_part(self, boundary_unscaled, shape_eig):
        """
        Calculate the variance of one dimension of the truncated part.
        :param boundary_unscaled: 2D unscaled boundary of the current dimension of the truncation area
        :param shape_eig: eigenvalue of the current dimension of the original Gaussians covariance matrix
        :return: variance of the current dimension of the truncated part
        """
        if (boundary_unscaled[0] == -np.inf) & (boundary_unscaled[1] == np.inf):
            # everything is truncated, so the truncated part is equal to the original Gaussian
            h_var = shape_eig
        elif boundary_unscaled[1] == np.inf:
            h_var = shape_eig * (1.0 + boundary_unscaled[0] * norm_pdf.pdf(boundary_unscaled[0])
                                 / (1.0 - norm_pdf.cdf(boundary_unscaled[0]))
                                 - (norm_pdf.pdf(boundary_unscaled[0]) / (
                            1.0 - norm_pdf.cdf(boundary_unscaled[0]))) ** 2)
        elif boundary_unscaled[0] == -np.inf:
            h_var = shape_eig * (1.0 - boundary_unscaled[1] * norm_pdf.pdf(boundary_unscaled[1])
                                 / (norm_pdf.cdf(boundary_unscaled[1]))
                                 - (-norm_pdf.pdf(boundary_unscaled[1]) / (norm_pdf.cdf(boundary_unscaled[1]))) ** 2)
        else:
            h_var = shape_eig * (((boundary_unscaled[0] * norm_pdf.pdf(boundary_unscaled[0])
                                   - boundary_unscaled[1] * norm_pdf.pdf(boundary_unscaled[1]))
                                  / (norm_pdf.cdf(boundary_unscaled[1]) - norm_pdf.cdf(boundary_unscaled[0])))
                                 - ((norm_pdf.pdf(boundary_unscaled[0]) - norm_pdf.pdf(boundary_unscaled[1]))
                                    / (norm_pdf.cdf(boundary_unscaled[1]) - norm_pdf.cdf(
                                boundary_unscaled[0]))) ** 2 + 1.0)

        return h_var

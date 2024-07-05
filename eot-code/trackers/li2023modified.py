"""
Implements a modified random matrix tracker for 2D LiDAR, proposed by Li et al. (2023)

Original article:
P. Li, C. Chen, C. You, and J. Qiu,
“Modified extended object tracker for 2D lidar data using random matrix model,”
Sci Rep, vol. 13, no. 1, Art. no. 1, Mar. 2023, doi: 10.1038/s41598-023-32236-w.
"""
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

from trackers.abstract_tracking import AbtractEllipticalTracker
from utility.unscented_transform import unscented_transform
from utility.utils import cart2pol_pt, pol2cart_pt


class Li2023ModifiedRM(AbtractEllipticalTracker):
    """
    Implements a modified random matrix tracker for 2D LiDAR, proposed by Li et al. (2023)
    """

    d = 2
    use_unscented_transform = True

    def __init__(self,
                 P_init,
                 v_init,
                 R,
                 F,
                 Q,
                 H,
                 nu,
                 n_lidar_scans_per_rotation,
                 m_init=None,
                 V_init=None,
                 A=None,
                 uniform_weights=True,
                 scale_parameter=0.25):
        self.R = R
        self.F = F
        self.m = m_init
        self.P = P_init

        self.A = A if A is not None else np.eye(self.d)  # "change mode" default to identity

        self.v = v_init
        self.V = V_init
        self.Q = Q[:4, :4]
        self.H = np.array(H) if H is not None else np.hstack([np.eye(2), np.zeros((2, len(self.P) - 2))])
        self.kappa = scale_parameter
        self.nu = nu
        self.n_last = None  # $n_{k-1}$ the number of measurements received in the last time step
        self.n_lidar_scans_per_rotation = n_lidar_scans_per_rotation
        self.uniform_weights = uniform_weights

    def standard_predict(self):
        # kinematics
        F = self.F
        self.m = F @ self.m
        self.P = F @ self.P @ F.T + self.Q
        tau = 2
        v_plus = np.exp(-1 / tau) * (self.v - 2 * self.d - 2)
        v_plus += 2 * self.d + 2
        V_plus = ((v_plus - self.d - 1) / (self.v - self.d - 1)) * self.V
        self.v = v_plus
        self.V = V_plus

        return self.get_state()

    def predict(self):
        # return self.standard_predict()
        # (13a)
        self.m = self.F @ self.m
        # (13b)
        self.P = self.F @ self.P @ self.F.T + self.Q
        # (13c)
        # n_last := $n_{k|k-1}
        v_minus = self.v
        self.v = self.d + 1 + (self.n_last * (self.v - self.d - 1)) / (self.n_last + self.v - 2 * self.d - 2)
        # (13d)
        # uses the variable "n" which is never defined - using the prior v similar to other methods
        self.V = (self.A @ self.V @ self.A.T) / (1 + (self.v - self.d - 1) / (v_minus - self.d - 1))

        return self.get_state()

    def _init_filter(self, measurements):
        zhat, Zhat = self._compute_zZ_hat(measurements)
        self.m = np.array([*zhat, 0, 0])
        self.V = Zhat
        return self.get_state()

    def _compute_segment_midpoints(self, measurements):
        r = np.arctan2(measurements[:, 1], measurements[:, 0]) % (2 * np.pi)
        segments = np.linspace(0, 2 * np.pi, self.n_lidar_scans_per_rotation)
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

        # add middle segment mean on its own (if uneven number of segments and central segment has no nan entry)
        if len(relevant_segments) % 2 == 1:
            try:
                central_segment_mean = relevant_segments[int(len(relevant_segments) / 2) + 1]
                if not np.isnan(central_segment_mean).any():
                    midpoints.append(central_segment_mean)
            except IndexError:  # skip
                pass
        self.datadict = dict(
            segments=segments,
            digitized=digitized,
            bin_means=bin_means,
            filled_segments=filled_segments,
            first_detected_segment=first_detected_segment,
            last_detected_segment=last_detected_segment,
            relevant_segments=relevant_segments,
        )
        return np.array(midpoints)

    def _compute_weights(self, z):
        n_k = len(z)
        if self.uniform_weights or z.shape[0] == 1:
            return np.full(shape=(n_k,), fill_value=1)
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

    def _compute_zZ_hat(self, measurements):
        """
        Implements equations (10) and (12) of the paper to acquire $\hat{z}$ and $\hat{Z}$
        """
        z = measurements
        n_k = len(z)

        omega = self._compute_weights(measurements)
        # Z_bar is the set of matched segment means
        Z_bar = self._compute_segment_midpoints(measurements)
        n_bar = len(Z_bar)
        if n_bar <= self.nu:
            z_hat = np.average(z, weights=omega, axis=0)
        else:
            A1, B1, A2, B2 = self.get_tangent_orthogonal_line_coefficients(measurements)

            # compute C
            C = np.ones((n_bar, 2))  # start with 1s everywhere
            C[:, 1] = Z_bar[:, 0]
            D = Z_bar[:, 1].reshape((n_bar, 1))
            # extract least squares solution for A/B_3
            # note that the order must be inverted compared to what is given in the paper
            B3, A3 = inv(C.T @ C) @ C.T @ D

            mu_1 = np.array([
                (B3 - B1) / (A1 - A3),
                A1 * ((B3 - B1) / (A1 - A3)) + B1
            ])
            mu_2 = np.array([
                (B3 - B2) / (A2 - A3),
                A2 * ((B3 - B2) / (A2 - A3)) + B2
            ])
            z_hat = np.mean([mu_1, mu_2], axis=0).reshape((-1,))

        Z_hat = np.sum([
            omega[i] * ((z[i] - z_hat).reshape((2, 1)) @ (z[i] - z_hat).reshape((2, 1)).T)
            for i in range(n_k)
        ], axis=0)

        return z_hat, Z_hat

    def _get_scaled_shape_matrix(self):
        """return the current shape matrix X_hat, scaled with kappa"""
        return self.kappa * self._get_X_hat()

    def _get_X_hat(self):
        # the acquisition of X_hat is not defined in Li et al, using Feldmann for that
        return self.V / (self.v - self.d - 1)

    def update(self, measurements: np.ndarray):
        self.n_last = len(measurements)
        n_k = len(measurements)
        if self.m is None or self.V is None:
            return self._init_filter(measurements)
        # ---
        # apply (10) and (12) to get $\hat{z}$ and $\hat{Z}$
        z_hat, Z_hat = self._compute_zZ_hat(measurements)
        # compute preliminary variables
        X_hat = self._get_X_hat()

        if self.use_unscented_transform:
            _, R, _ = unscented_transform(mu=cart2pol_pt(z_hat.reshape((-1,))), C=self.R, f=pol2cart_pt)
        else:
            R = self.R
        Y = self._get_scaled_shape_matrix() + R
        S = self.H @ self.P @ self.H.T + (Y / n_k)
        S_inv = inv(S)
        K = self.P @ self.H.T @ S_inv
        eps = z_hat - self.H @ self.m
        eps = eps.reshape((-1, 1))
        X_2 = np.array(sqrtm(X_hat)).astype(float)
        S_i2 = np.array(sqrtm(S_inv)).astype(float)
        Y_i2 = np.array(sqrtm(np.linalg.inv(Y))).astype(float)
        N_hat = X_2 @ S_i2 @ eps @ eps.T @ S_i2.T @ X_2.T
        Y_hat = X_2 @ Y_i2 @ Z_hat @ Y_i2.T @ X_2.T

        # computed updated state estimate
        m_plus = self.m + (K @ eps).reshape((-1,))
        P_plus = self.P + K @ S @ K.T
        v_plus = self.v + n_k
        V_plus = self.V + N_hat + Y_hat

        # save
        self.m = m_plus
        self.P = P_plus
        self.v = v_plus
        self.V = V_plus

        return self.get_state()

    def get_state(self):
        x, y = self.m.reshape((-1,))[:2]
        orientation, length, width = self._matrix_to_params_rm(self.V / (self.v - self.d - 1))
        # convert semi-axis length to axis length
        length, width = length, width
        velo_x, velo_y = self.m.reshape((-1,))[2:4]
        state = np.array([x, y, velo_x, velo_y, orientation, length, width]).astype(float)
        return state

    def set_R(self, R):
        assert R.shape == self.R.shape
        self.R = R

    @staticmethod
    def _matrix_to_params_rm(X):
        """Convert shape matrix X to parameter form [alpha, l1, l2] with semi-axis length"""
        assert X.shape == (2, 2), "X is not a 2x2 matrix"
        val, vec = np.linalg.eig(X)  # eigenvalue decomposition
        alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
        alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
        p = [alpha, *np.sqrt(val)]
        return np.array(p)

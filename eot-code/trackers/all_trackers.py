import numpy as np
from scipy.linalg import block_diag

from trackers.feldmann_random_matrix import FeldmannRMTracker
from trackers.li2023modified import Li2023ModifiedRM
from trackers.li23_feldmannLike import LiRMLikeFeldmann
from trackers.alqaderi_ekf import AlqaderiEKF
from trackers.truncated_random_matrix import TruncatedRMTracker
from trackers.memekf import MEMEKF
from trackers.ours_lidar_random_matrix import FeldmannAdaptedToLidar


def get_example_tracker(method_id: str,
                        lidar_kwargs,
                        Q,
                        use_UT=True,
                        tau=1,
                        v_init=20,
                        ):
    R_polar = lidar_kwargs["R"]
    n_r = lidar_kwargs["horizontal_resolution"]
    F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    R_cart = np.eye(2) * 0.2

    if use_UT:
        R = R_polar
    else:
        R = R_cart

    method_id = method_id.lower()
    Q_kinematic = Q[:4, :4]
    if method_id == "feldmann" or method_id == "rm":
        t = FeldmannRMTracker(P=np.diag([0.1, 0.1, 10, 10]),
                              v=v_init,
                              R=R,
                              Q=Q_kinematic,
                              time_step_length=1,
                              tau=tau,
                              z_scale=2 / 3)
    elif method_id == "lidarrm" or method_id == "ours":
        t = FeldmannAdaptedToLidar(P=np.diag([0.1, 0.1, 10, 10]),
                                   v=v_init,
                                   R=R,
                                   Q=Q_kinematic,
                                   time_step_length=1,
                                   tau=tau,
                                   z_scale=2 / 3,
                                   lidar_kwargs=lidar_kwargs)

    elif method_id == "li" or method_id == "li2023" or method_id == "li2023modified":
        t = Li2023ModifiedRM(P_init=np.diag([0.1, 0.1, 10, 10]),
                             v_init=v_init,
                             R=R,
                             F=F,
                             Q=Q_kinematic,
                             H=None,
                             nu=3,
                             n_lidar_scans_per_rotation=n_r,
                             A=None,
                             scale_parameter=2 / 3)
    elif method_id == "liadapted" or method_id == "liimproved":
        t = LiRMLikeFeldmann(P=np.diag([0.1, 0.1, 10, 10]),
                             v=v_init,
                             R=R,
                             Q=Q_kinematic,
                             time_step_length=1,
                             tau=tau,
                             nu=3,
                             z_scale=2 / 3,
                             lidar_kwargs=lidar_kwargs)
    elif method_id == "truncatedrm":
        t = TruncatedRMTracker(P=np.diag([0.1, 0.1, 10, 10]),
                               v=v_init,
                               R=R,
                               Q=Q_kinematic,
                               time_step_length=1,
                               tau=tau,
                               z_scale=1 / 3)
    elif method_id == "alqaderiekf":
        Q_no_theta = np.zeros((6, 6))
        # take kinematic process noise
        Q_no_theta[:4, :4] = Q[:4, :4]
        # take semi axis process noise
        Q_no_theta[-2:, -2:] = Q[-2:, -2:]
        t = AlqaderiEKF(P_init=np.diag([0.25, 0.25, 10, 10, 5, 5]),
                        R=R,
                        F=block_diag(F, np.eye(2)),
                        Q=Q_no_theta,
                        n_gmm_components=64)  # Fig.3 in their paper looks like 64 pts?
    elif method_id == 'memekf' or method_id == 'gm-memekf' or method_id == 'gm-memekf*' or method_id == 'gm-mem-ekf*':
        t = MEMEKF(cov_kin=np.diag([0.1, 0.1, 10, 10]),
                   cov_shape=np.diag([0.1, 0.5, 0.5]),
                   measurement_noise_cov_polar=R,
                   kin_process_cov=Q_kinematic,
                   shape_process_cov=np.diag([0.05 * np.pi, 0.02, 0.02]))
    else:
        raise ValueError(f"Unknown method '{method_id}'!")
    t.use_unscented_transform = use_UT
    return t

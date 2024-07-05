import numpy as np

from utility.data_generation import sample_cv_trajectory, run_tracker_on_measurements
from utility.metrics import full_state_iou


def get_single_run_iou_list(rng: np.random.Generator,
                            initial_ranges,
                            n_steps_per_run,
                            lidar_kwargs,
                            process_noise_covariance,
                            trackers: dict):
    iou_per_step = {t: [] for t in trackers.keys()}
    states, measurements = sample_cv_trajectory(initial_ranges=initial_ranges,
                                                rng=rng,
                                                n_steps=n_steps_per_run,
                                                Q=process_noise_covariance,
                                                match_heading_to_yaw=True,
                                                **lidar_kwargs
                                                )
    for tracker_id in trackers.keys():
        estimates = run_tracker_on_measurements(trackers[tracker_id], states, measurements)
        iou_per_step[tracker_id] = [full_state_iou(est, gt) for est, gt in zip(estimates, states)]

    return iou_per_step

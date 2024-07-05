from copy import deepcopy
from scipy.linalg import block_diag
from shapely.geometry import  LineString, Point

from utility.utils import get_shapely_rectangle, pol2cart, rot, state_to_rect_corner_pts
from utility.constants import *


def get_rect_contour_measurements(m, p,
                                  horizontal_resolution: int,
                                  sensor_range=200,
                                  R=None,
                                  rng=None,
                                  polar_noise=False,
                                  efficient=True):
    """
    Sample contour measurements from a rectangle and return (noisy) observations.
    :param m: kinematic state, i.e., position (as x,y)
    :param p: shape state as theta, length, width (semi-axis)
    :param horizontal_resolution: Number of scans performed. Resolution of the simulated lidar hence is 2pi / this value
    :param sensor_range: Maximum range measurements can be from the sensor (located at the origin) before they are not
    returned anymore
    :param R: Measurement noise covariance as 2x2 matrix or None to not apply any meas. noise.
    :param rng: numpy Generator used for random sampling
    :param polar_noise: Whether measurement noise should be applied in polar or Cartesian coordinates, i.e. whether
    the dims of R are (range, angle) or (x,y)
    :param efficient: If True, will try to efficiently sample for the given object instead of raycasting every single
    one of the horizontal_resolution many laser beams.
    :return: Nx2 array of (noisy) measurements from the target contour.
    """
    if R is not None:
        rng = rng if rng is not None else np.random.default_rng()
    sensor_location = np.array([0, 0])
    target = get_shapely_rectangle(m, p)

    if target.contains(Point(*sensor_location)):
        raise ValueError("Target encompasses sensor location")

    Z = []
    # determine angles at which lidar "fires"
    covered_yaw_angles = np.linspace(0, 2 * np.pi, horizontal_resolution, endpoint=False)

    # check if object is on x-Axis, in that case angle wrapping causes issues for the efficient calcs
    no_x_intersection = len(np.array(
        target.intersection(
            LineString(
                [sensor_location,
                 rot(0) @ np.asarray([sensor_range, 0])])
        ).coords)) == 0
    if efficient and no_x_intersection:
        # reduce angles to relevant subset
        corner_yaws = [
            np.arctan2(pt[1], pt[0]) % (2 * np.pi)
            for pt in state_to_rect_corner_pts(m, p)[:-1]  # skip last point since it's duplicate
        ]
        # eliminate yaw angles that would miss anyway
        covered_yaw_angles = covered_yaw_angles[covered_yaw_angles >= np.min(corner_yaws)]
        covered_yaw_angles = covered_yaw_angles[covered_yaw_angles <= np.max(corner_yaws)]
    for yaw in covered_yaw_angles:
        intersection = np.array(
            target.intersection(
                LineString(
                    [sensor_location,
                     rot(yaw) @ np.asarray([sensor_range, 0])])
            ).coords
        )
        # skip beams that did not hit a target
        if len(intersection) == 0:
            continue

        # find closer point of intersection (= facing sensor)
        if np.linalg.norm(intersection[0, :] - sensor_location) < np.linalg.norm(intersection[1, :] - sensor_location):
            next_point = intersection[0, :]
        else:
            next_point = intersection[1, :]

        if R is not None and polar_noise:
            next_point = pol2cart(*rng.multivariate_normal(
                np.asarray([np.linalg.norm(next_point), yaw]),
                R
            ))
        Z.append(next_point)

    if R is not None and not polar_noise:
        rng = rng if rng is not None else np.random.default_rng()
        Z = np.array([rng.multivariate_normal(y, R) for y in Z])

    # if len(Z) == 0:
    #     raise ValueError("Target didn't generate measurements")

    return np.array(Z)


def get_cv_trajectory(init_state: np.ndarray,  # 7d state array
                      rng: np.random.Generator,
                      n_steps: int,
                      Q: np.ndarray,
                      match_heading_to_yaw: bool,
                      **lidar_parameters
                      ):
    """
    Generate a tuple (states, measurements) for a given set of parameters, following a nearly constant velocity
    trajectory. Ground truth states and measurements are generated and returned.
    :param init_state: 7D initial state: [x, y, vel_x, vel_y, yaw, length, width]
    :param rng: random number generator object to use
    :param n_steps: number of steps to run the trajectory for
    :param Q: None or 7x7 process noise covariance. If None, process noise is disables
    :param match_heading_to_yaw: If True, the orientation of the object will be overwritten with the velocity heading
    in every time step
    :param lidar_parameters: Kwargs dict of lidar parameters passed to the measurement generation function, must include
         horizontal_resolution: Horizontal resolution of the sensor as integer number of beams fired per frame
         sensor_range: Maximum range of the sensor, length of raycast beams
         R: 2x2 measurement covariance. For polar noise, dims are (range, angle), else (x, y).
         polar_noise: Boolean indicating whether meas. noise is applied in polar coordinates
    :return: (states, measurements) of type (ndarray, list[ndarray]) of the trjaectory
    """
    # Define the motion model
    F_m = np.array([  # kinematic motion model: constant velocity state transition
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    F_p = np.eye(3)  # shape remains the constant
    F = block_diag(F_m, F_p)

    # generate trajectory by iterating using motion model, starting from the given init. state
    init_state = np.array(init_state)[:7]
    if match_heading_to_yaw:
        init_state[IX_YAW] = np.arctan2(init_state[IX_VEL_Y], init_state[IX_VEL_X])
    states = [init_state]
    for i in range(n_steps - 1):
        next_state = rng.multivariate_normal(F @ states[-1], Q) if Q is not None else F @ states[-1]
        if match_heading_to_yaw:
            next_state[IX_YAW] = np.arctan2(next_state[IX_VEL_Y], next_state[IX_VEL_X])
        states.append(next_state)
    # generate measurements for trajectory
    final_states = []
    measurements = []
    for state in states:
        next_measurement_set = get_rect_contour_measurements(m=state[IXS_LOCATiON],
                                                             p=state[IXS_SHAPE],
                                                             rng=rng,
                                                             **lidar_parameters)
        if len(next_measurement_set) == 0:
            break
        final_states.append(state)
        measurements.append(next_measurement_set)

    return np.array(final_states), measurements


def sample_cv_trajectory(initial_ranges: np.ndarray,
                         rng: np.random.Generator,
                         n_steps: int,
                         Q: np.ndarray,
                         match_heading_to_yaw: bool,
                         **lidar_parameters):
    """
    Create a constant velocity trajectory from a randomized initial guess.

    Uses a simple error-catching accept-reject scheme until a valid trajectory is found

    :param initial_ranges: 7x2 array of ranges from which initial position is sampled uniformly
    :param rng: random number generator object to use
    :param n_steps: number of steps to run the trajectory for
    :param Q: None or 7x7 process noise covariance. If None, process noise is disables
    :param match_heading_to_yaw: If True, the orientation of the object will be overwritten with the velocity heading
    in every time step
    :param lidar_parameters: Kwargs dict of lidar parameters passed to the measurement generation function, must include
         horizontal_resolution: Horizontal resolution of the sensor as integer number of beams fired per frame
         sensor_range: Maximum range of the sensor, length of raycast beams
         R: 2x2 measurement covariance. For polar noise, dims are (range, angle), else (x, y).
         polar_noise: Boolean indicating whether meas. noise is applied in polar coordinates
    :return: states, measurements tuple of CV trajectory
    """
    if initial_ranges.shape == (7, 2):
        initial_state = rng.uniform(initial_ranges[:, 0], initial_ranges[:, 1])
    elif initial_ranges.shape == (7,) or initial_ranges.shape == (7, 1):
        initial_state = np.array(initial_ranges)
    else:
        raise ValueError(f"Initial ranges are of shape {initial_ranges.shape} (must be 7x2 or 7x1 / 7d)!")
    success = False
    while not success:
        try:
            states, measurements = get_cv_trajectory(init_state=initial_state,
                                                     rng=rng,
                                                     n_steps=n_steps,
                                                     Q=Q,
                                                     match_heading_to_yaw=match_heading_to_yaw,
                                                     **lidar_parameters)
            success = True
        except ValueError as e:
            # trajectory got too close to sensor, repeat
            pass
    return states, measurements


def run_tracker_on_measurements(tracker, ground_truth, measurements):
    """
    Given a tracker, ground truth states and measurements, apply the tracker to all measurements sequentially and
    return the resulting state estimates as ndarray.
    """
    estimates = []
    for gt, Z in zip(ground_truth, measurements):
        if tracker.REQUIRES_GROUND_TRUTH:
            tracker.update(Z, ground_truth=gt)
        else:
            tracker.update(Z)
        estimates.append(tracker.get_state())
        tracker.predict()
    return np.array(estimates)


def generate_data_for_seed(seed,
                           trackers: dict,
                           initial_ranges,
                           n_steps_per_run,
                           process_noise_covariance,
                           lidar_kwargs
                           ):
    """
    Samples a trajectory for a given seed, then runs the set of trackers on the data, and returns a tuple of
    (ground_truth_states, generated_measurements, tracker_estimates) where the estimates follow the same dict structure
    as the input tracker dict (i.e. for each key in trackers, a corresponding key with estimates for this tracker
    exists)
    """
    rng = np.random.default_rng(seed)
    states, measurements = sample_cv_trajectory(initial_ranges=initial_ranges,
                                                rng=rng,
                                                n_steps=n_steps_per_run,
                                                Q=process_noise_covariance,
                                                match_heading_to_yaw=True,
                                                **lidar_kwargs
                                                )
    estimates = {
        tracker_id: run_tracker_on_measurements(deepcopy(trackers[tracker_id]), states, measurements)
        for tracker_id in trackers.keys()
    }

    return states, measurements, estimates


def get_tracker_estimates_on_trajectory(states,
                                        measurements,
                                        trackers,
                                        ):
    estimates = {
        tracker_id: run_tracker_on_measurements(deepcopy(trackers[tracker_id]), states, measurements)
        for tracker_id in trackers.keys()
    }

    return estimates

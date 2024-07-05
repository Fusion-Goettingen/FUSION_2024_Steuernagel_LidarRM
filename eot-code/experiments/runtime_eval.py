import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from utility.data_generation import sample_cv_trajectory
from utility.constants_experiments import *
from utility.constants import *
from trackers.all_trackers import *


def result_generation(fp):
    lidar_kwargs = OS1_DEFAULT
    n_steps_per_run = TRAJECTORY_LENGTH
    runtime_matrix_dict = {t: [] for t in TRACKER_NAME_LIST}
    trackers = {t: get_example_tracker(method_id=t,
                                       lidar_kwargs=lidar_kwargs,
                                       Q=PROCESS_NOISE_COVARIANCE_MATRIX,
                                       use_UT=True,
                                       tau=1,
                                       v_init=20)
                for t in TRACKER_NAME_LIST}
    seed = np.random.default_rng(EXPERIMENT_SEED).integers(low=1, high=9999999999, size=N_RUNS_RUNTIME)
    for run_ix in tqdm(range(N_RUNS_RUNTIME)):
        states, measurements = sample_cv_trajectory(initial_ranges=INITIAL_RANGES,
                                                    rng=np.random.default_rng(seed[run_ix]),
                                                    n_steps=n_steps_per_run,
                                                    Q=PROCESS_NOISE_COVARIANCE_MATRIX,
                                                    match_heading_to_yaw=True,
                                                    **lidar_kwargs
                                                    )
        for t in TRACKER_NAME_LIST:
            for gt, Z in zip(states, measurements):
                if trackers[t].REQUIRES_GROUND_TRUTH:
                    t0 = time.time()
                    trackers[t].update(Z, ground_truth=gt)
                    t_update = time.time() - t0
                else:
                    t0 = time.time()
                    trackers[t].update(Z)
                    t_update = time.time() - t0
                t0 = time.time()
                trackers[t].predict()
                t_predict = time.time() - t0
                runtime_matrix_dict[t].append([t_update + t_predict, t_update, t_predict])
    with open(fp, "w") as f:
        json.dump(runtime_matrix_dict, f, indent=4)


def analysis(fp, fix_style=False, ignore_parts=None):
    if fix_style:
        plt.style.use(STYLE_SHEET)

    with open(fp, "r") as f:
        runtime_dict = json.load(f)

    runtime_dict_ms = {t: np.array(runtime_dict[t]) * 1000 for t in runtime_dict.keys()}
    trackers = list(runtime_dict_ms.keys())
    if ignore_parts is not None:
        trackers = [t for t in trackers
                    if not np.any([p in t.lower() for p in ignore_parts])]
    # each runtime dict entry is of shape (n_steps_in_all_traj, 3) where 3 is for (total, predict, update)
    print(f"Overall mean runtimes of {len(runtime_dict_ms[list(runtime_dict_ms.keys())[0]])} time steps")
    for t_id in trackers:
        print(f"\t{t_id}: {runtime_dict_ms[t_id][:, 0].mean():.2f}ms")

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))
    barplot = plt.bar(trackers, [runtime_dict_ms[t_id][:, 0].mean() for t_id in trackers])
    plt.gca().bar_label(barplot, labels=[f'{x:.2f}' for x in barplot.datavalues])
    plt.ylim(0, plt.ylim()[1] * 1.05)  # upscale y-axis to ensure all text actually fits in the figure
    plt.ylabel("Mean runtime / ms")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    output_file_path = "../../output/paper/runtime_results.json"
    result_generation(output_file_path)
    analysis(output_file_path, fix_style=False)

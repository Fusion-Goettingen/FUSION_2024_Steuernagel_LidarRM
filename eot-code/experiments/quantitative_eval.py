import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from utility.constants_experiments import *
from utility.evaluation import get_single_run_iou_list
from utility.constants import *
from trackers.all_trackers import *

RENAME_DICT = {  # str -> str: rename key to value
    "LiAdapted": "LiImproved",
    "GM-MEMEKF*": "GM-MEM-EKF*"
}

FIGSIZE = (25, 6)  # (22, 7)


def load_dict(fp):
    with open(fp, "r") as f:
        iou_matrix_dict = json.load(f)

    if RENAME_DICT is not None:
        def rename_key(key):
            if key in RENAME_DICT.keys():
                return RENAME_DICT[key]
            else:
                return key

        corrected_iou_matrix_dict = {
            rename_key(k): iou_matrix_dict[k]
            for k in iou_matrix_dict.keys()
        }
        return corrected_iou_matrix_dict
    else:
        return iou_matrix_dict


def result_generation(fp, lidar_kwargs=OS1_EASY, tracker_names=None):
    if tracker_names is None:
        tracker_names = TRACKER_NAME_LIST
    n_steps_per_run = TRAJECTORY_LENGTH
    iou_matrix_dict = {t: [] for t in tracker_names}
    seed_list = np.random.default_rng(EXPERIMENT_SEED).integers(low=1, high=9999999, size=N_MONTE_CARLO_RUNS)
    for run_ix in tqdm(range(N_MONTE_CARLO_RUNS)):
        iou_list_dict = get_single_run_iou_list(np.random.default_rng(seed_list[run_ix]), initial_ranges=INITIAL_RANGES,
                                                n_steps_per_run=n_steps_per_run, lidar_kwargs=lidar_kwargs,
                                                process_noise_covariance=PROCESS_NOISE_COVARIANCE_MATRIX,
                                                trackers={t: get_example_tracker(method_id=t,
                                                                                 lidar_kwargs=lidar_kwargs,
                                                                                 Q=PROCESS_NOISE_COVARIANCE_MATRIX,
                                                                                 use_UT=True,
                                                                                 tau=1,
                                                                                 v_init=20)
                                                          for t in tracker_names})
        for t in tracker_names:
            iou_matrix_dict[t].append(iou_list_dict[t])

    with open(fp, "w") as f:
        json.dump(iou_matrix_dict, f, indent=4)


def flatten(xss):
    return [x for xs in xss for x in xs]


def perform_results_analysis(fp, fixed_style=False, disable_visuals=False):
    if fixed_style:
        plt.style.use(STYLE_SHEET)
    plt.rcParams["font.size"] = 26

    iou_matrix_dict = load_dict(fp)
    print(f"Loaded {fp}")
    tracker_name_list = list(iou_matrix_dict.keys())
    flat_iou_matrix_dict = {t: np.array(flatten(iou_matrix_dict[t])) for t in tracker_name_list}

    # === PRINT BASED EVAL
    # Means
    print("Overall Mean IoU:")
    best_mean_iou = np.max([np.mean(flat_iou_matrix_dict[t]) for t in tracker_name_list])
    for t in tracker_name_list:
        percent_best = (np.mean(flat_iou_matrix_dict[t]) / best_mean_iou) * 100
        print(f"\t{t}: "
              f"{np.mean(flat_iou_matrix_dict[t]):.3f} "
              f"± {np.std(flat_iou_matrix_dict[t]):.3f} "
              f"({percent_best:.1f}%)")

    print("")

    if not disable_visuals:
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)

        off = 0.3
        for i, t in enumerate(tracker_name_list):
            all_data = flat_iou_matrix_dict[t].flatten()
            mean = np.average(all_data)
            median = np.median(all_data)
            std = np.std(all_data)

            ds = 0.75

            # mean, median
            plt.plot([i - off * ds, i + off * ds], [median, median],
                     linestyle=':',
                     c='darkgreen',
                     )
            plt.plot([i - off, i + off], [mean, mean],
                     c='orange',
                     )

            # box
            plt.plot([i, i], [mean - std, mean + std],
                     c='k')
            plt.plot([i - off * ds, i + off * ds], [mean - std, mean - std],
                     c='k')
            plt.plot([i - off * ds, i + off * ds], [mean + std, mean + std],
                     c='k')
            plt.gca().add_patch(Rectangle(
                (i - off * ds, mean - std),
                2 * off * ds,
                std * 2,
                color='grey',
                fill=True,
                alpha=0.3
            ))

        plt.ylabel("IoU")
        xticklist = [f"{t}\n{np.mean(flat_iou_matrix_dict[t]):.2f}±{np.std(flat_iou_matrix_dict[t]):.2f}"
                     for t in tracker_name_list]
        plt.gca().set_xticks(list(range(len(tracker_name_list))),
                             xticklist)
        if not fixed_style:
            plt.title(f"IoU over all {N_MONTE_CARLO_RUNS} Monte Carlo Runs")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    disable_visuals = False
    settings = [
        ["../../output/paper/quantitative_results_OS1default.json", OS1_DEFAULT],
        ["../../output/paper/quantitative_results_OS1sparse.json", OS1_SPARSE],
    ]
    for settings_tuple in settings:
        output_file_path, lidar_kwargs = settings_tuple
        result_generation(fp=output_file_path, lidar_kwargs=lidar_kwargs)

    for settings_tuple in settings:
        output_file_path, lidar_kwargs = settings_tuple
        perform_results_analysis(output_file_path, fixed_style=True, disable_visuals=disable_visuals)

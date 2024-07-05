import matplotlib.pyplot as plt
from utility.constants_experiments import *
from utility.constants import *
from trackers.all_trackers import *
from utility.visuals import plot_trajectory, show_plot, add_sensor, get_color_cycle, plot_tracker, \
    plot_rectangular_state
from utility.data_generation import generate_data_for_seed

USE_FIXED_START = False


def visualize_set_of_trajectories(seed_list, tracker_names, delay=None, maintain_old_plot=False):
    lidar_kwargs = OS1_EASY
    trackers = {t: get_example_tracker(method_id=t,
                                       lidar_kwargs=lidar_kwargs,
                                       Q=PROCESS_NOISE_COVARIANCE_MATRIX,
                                       use_UT=True,
                                       tau=1,
                                       v_init=20)
                for t in tracker_names}
    init_pose = FIX_INITIAL_POSE if USE_FIXED_START else INITIAL_RANGES
    for seed in seed_list:
        states, measurements, estimates = generate_data_for_seed(seed=seed,
                                                                 trackers=trackers,
                                                                 initial_ranges=init_pose,
                                                                 n_steps_per_run=150,
                                                                 process_noise_covariance=PROCESS_NOISE_COVARIANCE_MATRIX,
                                                                 lidar_kwargs=lidar_kwargs)

        colors = get_color_cycle()
        if delay is None:
            plot_trajectory(states,
                            measurements,
                            label="Ground Truth",
                            fill=True,
                            state_color="grey",
                            state_alpha=0.5,
                            meas_color='k',
                            meas_alpha=1,
                            mark_initial=False)
            for i, tracker_id in enumerate(trackers.keys()):
                plot_tracker(estimates[tracker_id],
                             fill=False,
                             state_color=colors[i % len(colors)],
                             state_alpha=1,
                             label=tracker_id)
            add_sensor()
            show_plot()
        else:
            for time_ix in range(len(states)):
                if time_ix % VISUALIZE_EVERY_N_STEPS != 0:
                    continue
                if not maintain_old_plot:
                    plt.cla()
                x = states[time_ix]
                Z = measurements[time_ix]
                plot_rectangular_state(m=x[IXS_LOCATiON],
                                       p=x[IXS_SHAPE],
                                       fill=True,
                                       label="Ground Truth" if not maintain_old_plot or time_ix == 0 else None,
                                       color='grey',
                                       alpha=0.5)
                plt.scatter(*Z.T,
                            color='0.5',
                            alpha=1,
                            marker='.'
                            )
                for tracker_ix, tracker_id in enumerate(trackers.keys()):
                    est = estimates[tracker_id][time_ix]
                    plot_rectangular_state(m=est[IXS_LOCATiON],
                                           p=est[IXS_SHAPE],
                                           fill=False,
                                           label=tracker_id if not maintain_old_plot or time_ix == 0 else None,
                                           color=colors[tracker_ix % len(colors)],
                                           alpha=1)
                plt.scatter(0, 0, marker='d', c='k',
                            label='Sensor' if not maintain_old_plot or time_ix == 0 else None)
                plt.axis('equal')
                plt.xlabel(r"$m_1$ / m")
                plt.ylabel(r"$m_2$ / m")
                plt.legend()
                plt.tight_layout()
                plt.draw()
                plt.pause(delay)
            plt.show()


def feldmann_comp(seed_list, tracker_names, step_ix_list=None):
    plt.rcParams["font.size"] = 26
    lidar_kwargs = OS1_EASY
    trackers = {t: get_example_tracker(method_id=t,
                                       lidar_kwargs=lidar_kwargs,
                                       Q=PROCESS_NOISE_COVARIANCE_MATRIX,
                                       use_UT=True,
                                       tau=1,
                                       v_init=20)
                for t in tracker_names}
    init_pose = INITIAL_RANGES
    for seed in seed_list:
        states, measurements, estimates = generate_data_for_seed(seed=seed,
                                                                 trackers=trackers,
                                                                 initial_ranges=init_pose,
                                                                 n_steps_per_run=150,
                                                                 process_noise_covariance=PROCESS_NOISE_COVARIANCE_MATRIX,
                                                                 lidar_kwargs=lidar_kwargs)
        print(f"Inital step: {np.around(states[0], 4)}")
        if step_ix_list is not None:
            if len(step_ix_list) > len(states):
                print(f"Trying to access {len(step_ix_list)} steps, but only generated {len(states)}, "
                      f"using the first ones")
                step_ix_list = step_ix_list[:len(states)]
            states = np.array(states)[step_ix_list]
            measurements = np.array(measurements, dtype=object)[step_ix_list]
            for key in trackers.keys():
                estimates[key] = np.array(estimates[key])[step_ix_list]

        colors = get_color_cycle()
        plot_trajectory(states,
                        measurements,
                        label="Ground Truth",
                        fill=True,
                        state_color="grey",
                        state_alpha=0.5,
                        meas_color='k',
                        meas_alpha=1,
                        mark_initial=False)
        for i, tracker_id in enumerate(trackers.keys()):
            plot_tracker(estimates[tracker_id],
                         fill=False,
                         state_color=colors[i % len(colors)],
                         state_alpha=1,
                         label=tracker_id)
        plt.ylim(23, 50)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.axis('equal')
        plt.xlabel(r"$m_1$ / m")
        plt.ylabel(r"$m_2$ / m")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    plt.style.use(STYLE_SHEET)
    feldmann_comp(seed_list=[500], tracker_names=["Feldmann", "Ours"],
                  step_ix_list=np.arange(-6 * VISUALIZE_EVERY_N_STEPS, 0))  # grab last 6 steps

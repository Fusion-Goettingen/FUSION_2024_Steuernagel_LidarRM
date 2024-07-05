import numpy as np
import matplotlib.pyplot as plt

from utility.constants import STYLE_SHEET
from utility.naive_data_generation import get_rectangular_measurements
from utility.shape_estimation import mean_shifted_scattering_matrix_shape_estimate
from utility.visuals import plot_elliptic_extent, plot_rectangular_state


def plot_ellipse_for_scaling(m, scaling_factor, data, color_params):
    p_hat = mean_shifted_scattering_matrix_shape_estimate(Z=data,
                                                          m=None,
                                                          normalize=True,
                                                          scaling_factor=scaling_factor
                                                          )
    # convert axis lengths
    p_hat[1:] *= 2
    plot_elliptic_extent(m, p_hat, **color_params)


def visualize_scaling_factor(seed=42):
    m = [0, 0]
    p = [0, 4.7 / 2, 1.8 / 2]  # semi-axis
    n_meas = 200
    R = np.eye(2) * 0.1

    rng = np.random.default_rng(seed)
    measurement_sources, noisy_measurements = get_rectangular_measurements(
        loc=m,
        length=p[1] * 2,
        width=p[2] * 2,
        theta=p[0],
        n_measurements=n_meas,
        R=R,
        weight_list=[0.25, 0.25, 0.25, 0.25],
        internal_RNG=rng
    )

    # plot results
    plot_ellipse_for_scaling(m, 1 / 4, data=measurement_sources,
                             color_params={"label": r"$c = \frac{1}{4}$",
                                           "color": "green"})
    plot_ellipse_for_scaling(m, 1, data=measurement_sources,
                             color_params={"label": r"$c = 1$",
                                           "color": "orange"})
    plot_ellipse_for_scaling(m, 2 / 3, data=measurement_sources,
                             color_params={"label": r"$c = \frac{2}{3}$",
                                           "color": "blue"})

    # baseline data plot
    plt.scatter(*measurement_sources.T, c='k', zorder=3, marker='.', label='Measurements')
    plot_rectangular_state(m, p, fill=True, c='grey', alpha=0.4, zorder=2)
    plt.axis('equal')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))
    plt.legend()
    plt.tight_layout()
    plt.xlabel(r"$m_1$ / m")
    plt.ylabel(r"$m_2$ / m")
    plt.show()


def visualize_meanshift(seed=42):
    plt.rcParams["font.size"] = 26
    m = np.array([0, 0])
    p = [0, 4.7 / 2, 1.8 / 2]  # semi-axis
    n_meas = 200
    R = np.eye(2) * 0.1

    rng = np.random.default_rng(seed)
    measurement_sources, noisy_measurements = get_rectangular_measurements(
        loc=m,
        length=p[1] * 2,
        width=p[2] * 2,
        theta=p[0],
        n_measurements=n_meas,
        R=R,
        weight_list=[0.5, 0.5, 0, 0],
        internal_RNG=rng
    )

    # plot results
    p_mean = mean_shifted_scattering_matrix_shape_estimate(Z=measurement_sources,
                                                           m=None,
                                                           normalize=True,
                                                           scaling_factor=2 / 3
                                                           )
    p_shifted = mean_shifted_scattering_matrix_shape_estimate(Z=measurement_sources,
                                                              m=m,
                                                              normalize=True,
                                                              scaling_factor=2 / 3
                                                              )
    # convert axis lengths
    p_mean[1:] *= 2
    p_shifted[1:] *= 2
    plot_elliptic_extent(m, p_shifted, label="Center and adapted scatter matrix")
    plot_elliptic_extent(np.mean(measurement_sources, axis=0), p_mean,
                         label="Mean and scatter matrix", color='red')

    # baseline data plot
    plt.scatter(*measurement_sources.T, c='k', zorder=3, marker='.', label='Measurements')
    plot_rectangular_state(m, p, fill=True, c='grey', alpha=0.4, zorder=2)
    plt.ylim(-1.2, 2.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel(r"$m_1$ / m")
    plt.ylabel(r"$m_2$ / m")
    # plt.savefig(r"meanshift.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plt.style.use(STYLE_SHEET)
    visualize_scaling_factor()
    visualize_meanshift()

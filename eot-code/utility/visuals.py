import matplotlib.pyplot as plt
from matplotlib import patches
from utility.utils import rot, state_to_rect_corner_pts
from utility.constants import *
from utility.constants_experiments import VISUALIZE_EVERY_N_STEPS


def plot_elliptic_extent(m, p, ax=None, color='b', alpha=1., label=None, linestyle=None, show_center=True, fill=False,
                         show_orientation=False):
    """
    Add matplotlib ellipse patch based on location and extent information about vehicle
    :param m: Kinematic information as 4D array [x, y, velocity_x, velocity_y]
    :param p: extent information as 3D array [orientation, length, width]. Orientation in radians.
    :param ax: matplotlib axis to plot on or None (will use .gca() if None)
    :param color: Color to plot the ellipse and marker in
    :param alpha: Alpha value for plot
    :param label: Label to apply to plot or None to not add a label
    :param linestyle: Linestyle parameter passed to matplotlib
    :param show_center: If True, will additionally add an x for the center location
    :param fill: Whether to fill the ellipse
    """
    if ax is None:
        ax = plt.gca()
    theta, l1, l2 = p
    theta = np.rad2deg(theta)
    # patches.Ellipse takes angle counter-clockwise
    el = patches.Ellipse(xy=m[:2], width=l1, height=l2, angle=theta, fill=fill, color=color, label=label,
                         alpha=alpha, linestyle=linestyle)
    if show_center:
        ax.scatter(m[0], m[1], color=color, marker='x')

    if show_orientation:
        direction_vector = rot(p[0]) @ np.array([l1 / 2, 0]) + m[:2]
        ax.plot([m[0], direction_vector[0]], [m[1], direction_vector[1]], color=color)

    ax.add_patch(el)


def plot_rectangular_state(m, p,
                           fill=False,
                           **kwargs):
    """
    Plot a single state of a rectangular object
    :param m: kinematic state as x,y
    :param p: shape state as theta, l, w (semi-axis)
    :param fill: boolean indicating whether to to fill the rectangle (True) or just draw the outline (False)
    :param kwargs: additional keyword args passed on to plt.plot/plt.fill
    """
    m = np.array(m).reshape((-1,))
    p = np.array(p).reshape((-1,))
    if fill:
        plt.fill(*state_to_rect_corner_pts(m, p).T,
                 **kwargs)
    else:
        plt.plot(*state_to_rect_corner_pts(m, p).T,
                 **kwargs)


def plot_trajectory(states,
                    measurements,
                    label,
                    fill,
                    state_color,
                    state_alpha,
                    meas_color,
                    meas_alpha,
                    mark_initial=False,
                    ):
    """
    Plot a full trajectory consisting of states and measurements.
    Formatting of the plot and plt.show() are not included.
    :param states: List of tracker states as Nx7 ndarray (see constants.py for order)
    :param measurements: list of Nx2 ndarrays containing measurements corresponding to the individual track states.
    length of measurements and states must be equal.
    :param label: Optional label to add, or None to not label the track
    :param fill: boolean indicating whether shapes should be filled or not
    :param state_color: mpl color string for the state
    :param state_alpha: mpl opacity (alpha) value
    :param meas_color: color used for scatter plot of measurements
    :param meas_alpha: alpha value used for scatter plot of measurements
    :param mark_initial: If True, the first state will be marked by an X throughout a non-filled plotted version of
    the state. indicating where the trajectory started
    """
    assert len(states) == len(measurements)

    if mark_initial:
        corners = state_to_rect_corner_pts(states[0][IXS_LOCATiON], states[0][IXS_SHAPE])[:-1]
        for point_pair in [np.vstack([corners[0], corners[2]]), np.vstack([corners[1], corners[3]])]:
            plt.plot(*point_pair.T,
                     c=state_color,
                     alpha=1)

    for i in range(len(states)):
        if i % VISUALIZE_EVERY_N_STEPS != 0:
            continue
        x = states[i]
        Z = measurements[i]
        plot_rectangular_state(m=x[IXS_LOCATiON],
                               p=x[IXS_SHAPE],
                               fill=fill,
                               label=label if i == 0 else None,
                               color=state_color,
                               alpha=state_alpha)
        plt.scatter(*Z.T,
                    color=meas_color,
                    alpha=meas_alpha,
                    marker='.'
                    )


def show_plot(file=None):
    """
    Prepares visualization of the plot and either shows it or saves to file if given
    :param file: None to call plt.show or path to file in which plot should be saved
    """
    plt.axis('equal')
    plt.xlabel(r"$m_1$ / m")
    plt.ylabel(r"$m_2$ / m")
    plt.legend()
    plt.tight_layout()
    if file is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(file)


def add_sensor():
    """
    Adds the default visualization of the sensor to the current plot via a plt.scatter call (Sensor is located at 0,0)
    """
    plt.scatter(0, 0, marker='d', c='k', label='Sensor')


def plot_tracker(states,
                 fill,
                 state_color,
                 state_alpha,
                 label=None):
    """
    Plot all states of a given tracker as rectangles. A label can be added to the track.
    The global VISUALIZE_EVERY_N_STEPS will be used to determine whether every single state is plotted or not.
    :param states: List of tracker states as Nx7 ndarray (see constants.py for order)
    :param fill: boolean indicating whether shapes should be filled or not
    :param state_color: mpl color string for the state
    :param state_alpha: mpl opacity (alpha) value
    :param label: Optional label to add, or None to not label the track
    """
    for i in range(len(states)):
        if i % VISUALIZE_EVERY_N_STEPS != 0:
            continue
        x = states[i]
        plot_rectangular_state(m=x[IXS_LOCATiON],
                               p=x[IXS_SHAPE],
                               fill=fill,
                               label=label if i == 0 else None,
                               color=state_color,
                               alpha=state_alpha)


def get_color_cycle():
    """Return the matplotlib default color cycle as an array of strings representing colors"""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    return prop_cycle.by_key()['color']

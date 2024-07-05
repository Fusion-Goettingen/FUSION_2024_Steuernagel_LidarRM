from shapely.geometry import Polygon

from utility.utils import state_to_rect_corner_pts
from utility.constants import *


def iou(o1, o2):
    """
    Given two states as tuples (m, p) where the first two entries of m are the location and p contains orientation and
    semi-axis lengths, compute their intersection over union.

    :param o1: First object state as tuple as described above
    :param o2: Second object state as tuple as described above
    :return: IoU between objects
    """
    try:
        p1 = Polygon(state_to_rect_corner_pts(m=o1[0][:2], p=o1[1]))
        p2 = Polygon(state_to_rect_corner_pts(m=o2[0][:2], p=o2[1]))
        return p1.intersection(p2).area / p1.union(p2).area
    except:
        return 0


def full_state_iou(s0, s1):
    return iou(
        (s0[IXS_LOCATiON], s0[IXS_SHAPE]),
        (s1[IXS_LOCATiON], s1[IXS_SHAPE])
    )

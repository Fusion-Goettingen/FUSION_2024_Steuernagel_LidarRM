"""
Contains general constants.
"""
import numpy as np

# .mplstyle sheet to use for the generation of the plots
STYLE_SHEET = "../../data/paper.mplstyle"

# 7D state indices
IX_POS_X = 0
IX_POS_Y = 1
IX_VEL_X = 2
IX_VEL_Y = 3
IX_YAW = 4
IX_LENGTH = 5
IX_WIDTH = 6
IXS_LOCATiON = [IX_POS_X, IX_POS_Y]
IXS_SHAPE = [IX_YAW, IX_LENGTH, IX_WIDTH]

# lidar_kwargs: Keyword dicts describing different potential lidar settings
"""
LiDAR kwargs
     horizontal_resolution: Horizontal resolution of the sensor as integer number of beams fired per frame
     sensor_range: Maximum range of the sensor, length of raycast beams
     R: 2x2 measurement covariance. For polar noise, dims are (range, angle), else (x, y).
     polar_noise: Boolean indicating whether meas. noise is applied in polar coordinates
 
c.f. e.g.   
    https://data.ouster.io/downloads/datasheets/datasheet-revd-v2p0-os1.pdf
    Data Sheet of the Ouster OS-1 LiDAR
"""
# fix range of the sensor
_RANGE_OS1 = 100
OS1_DEFAULT = {
    "horizontal_resolution": 1024,
    "sensor_range": _RANGE_OS1,
    "R": np.diag([0.1 ** 2, np.deg2rad(0.01) ** 2]),
    "polar_noise": True
}

OS1_SPARSE = {
    "horizontal_resolution": 512,
    "sensor_range": _RANGE_OS1,
    "R": np.diag([0.1 ** 2, np.deg2rad(0.01) ** 2]),
    "polar_noise": True
}

OS1_EASY = {
    "horizontal_resolution": 2048,
    "sensor_range": _RANGE_OS1,
    "R": np.diag([0.03 ** 2, np.deg2rad(0.01) ** 2]),
    "polar_noise": True
}

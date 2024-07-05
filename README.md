# Random Matrix-based Tracking of Rectangular Extended Objects with Contour Measurements

Code for the [FUSION 2024](https://fusion2024.org/) paper:

```
Simon Steuernagel, Kolja Thormann, and Marcus Baum
Random Matrix-based Tracking of Rectangular Extended Objects with Contour Measurements
2024 27th International Conference on Information Fusion (FUSION). IEEE, 2024
```

**Abstract:**
A widely-used approach for extended object tracking is based on random matrices, where the scattering matrix, i.e.,
measurement spread, is used to update a symmetric positive definite random matrix representing an elliptic extent.
However, for lidar data, a mismatch between the assumed measurement model and observed data hinders the estimation
quality of the method.
We propose adaptions to the random matrix approach in order to facilitate the application for tracking a rectangular
extended object based on contour measurements.
Specifically, we derive a suitable scaling factor for the scattering matrix of measurements in this setting.
Furthermore, we propose a simple yet effective estimation scheme for the target center, adapting the shape estimate
accordingly.
The resulting algorithm closely follows the framework of the random matrix approach.
A detailed comparison with a variety of state-of-the-art trackers is carried out in a simulation based on real-world
lidar parameters, confirming the effectiveness of the approach.

Source code for the tracker can be found in [here](./eot-code/trackers/ours_lidar_random_matrix.py)

Scripts for reproducing figures from the papers:

- [Fig. 1](./eot-code/experiments/example_showcase_figure.py)
- [Fig. 2](./eot-code/experiments/scaling_factor_visualization.py)
- [Fig. 3](./eot-code/experiments/scaling_factor_visualization.py)
- [Fig. 4](./eot-code/experiments/quantitative_eval.py)
- [Table 1](./eot-code/experiments/runtime_eval.py)
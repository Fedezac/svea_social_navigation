# svea_social_navigation

This repository contains the code related to the `social_navigation` software stack, based on `MPC` and `SFM`.

## Installation
Please refer to svea on how to install and run SVEA software.\\
Make sure to build in the same workspace [PedSim](https://github.com/srl-freiburg/pedsim_ros) and that the [ROS Navigation Stack](https://github.com/ros-planning/navigation) is installed.

## Usage
For simulations, launch one the following files, accordingly to the scenario that the user wants:
```bash
roslaunch svea_social_navigation corridor_social.launch
```
```bash
roslaunch svea_social_navigation narrow_corridor_social.launch
```
```bash
roslaunch svea_social_navigation square_social.launch
```
For real-world experiments, make sure to run
```bash
roslaunch svea_social_navigation sml_social_real.launch
```
on a laptop (with the corresponding map), as well as `lli.launch` on the svea.


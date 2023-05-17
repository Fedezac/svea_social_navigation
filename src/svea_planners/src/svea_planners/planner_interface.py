#! /usr/bin/env python3

import rospy
import numpy as np
from svea_planners.gridmap_interface import GridMapInterface
from svea_planners.path_interface import PathInterface
from svea_planners.path_smoother import PathSmoother
from svea_planners.astar import AStarPlanner, AStarWorld
from nav_msgs.msg import Path

def assert_points(pts):
    assert isinstance(pts[0], (int, float)), 'points contain a coordinate pair wherein one value is not a number'
    assert isinstance(pts[1], (int, float)), 'points contain a coordinate pair wherein one value is not a number'
        
class PlannerInterface(object):
    _gridmap_interface = None
    _path_interface = None
    _world = None
    _planner = None
    _start = None
    _goal = None
    _path = None
    _obs_margin = None
    _social_waypoints = None

    def __init__(self, obs_margin=0.05, degree=3, s=None, theta_threshold=0.03):
        self._obs_margin = obs_margin

        # Path smoothing stuff
        self._degree = degree
        self._smoothing_parameter = s

        # Social waypoint extraction
        self._theta_threshold = theta_threshold

        self._gridmap_interface = GridMapInterface()
        self._gridmap_interface.init_ros_subscribers()
        rospy.loginfo("Planning interface correctly initialized")
        
    def initialize_planner_world(self):
        """
        Function that defines the planner object and the planner world 
        (which includes a grid representing the environment with its obstacles).
        It defines the planner only in case start and goal position are given.
        """
        delta, limits, obstacles = self._gridmap_interface.get_planner_world()
        self._world = AStarWorld(delta=delta, limit=limits, obstacles=np.multiply([delta[0], delta[1], 1], np.array(obstacles)).tolist(), obs_margin=self._obs_margin)
        if self._start and self._goal:
            self._planner = AStarPlanner(self._world, self._start, self._goal)
        else:
            raise Exception('No start and goal points were given')

    def compute_path(self):
        self._path = self._planner.create_path()

    def get_path_from_topic(self):
        msg = rospy.wait_for_message('/smooth_path', Path, timeout=None)
        path = []
        for p in msg.poses:
            path.append([p.pose.position.x, p.pose.position.y])
        self._path = path
        return path

    def initialize_path_interface(self):
        if self._path is not None:
            self._path_interface = PathInterface(self._path)
        else:
            raise Exception('No path was computed beforehand')
        
    def publish_rviz_path(self):
        self._path_interface.create_pose_path()
        self._path_interface.publish_rviz_path()

    def get_points_path(self, granularity=None, interpolate=False):
        """
        Function to get every (x, y) point composing the path

        :param granularity: get one point of the path each N-granularity points, defaults to None
        :type granularity: integer, optional
        :param interpolate: choose if path smoother has to interplate or approximate path, defaults to False
        :type iterpolate: boolean
        :return: list of points
        :rtype: list[float]
        """
        if granularity is not None:
            path = self._path_interface.get_points_path_reduced(granularity)
        else:
            path = self._path_interface.get_points_path()
        self._path_smoother = PathSmoother(path)
        if interpolate:
            return np.array(self._path_smoother.interpolate_b_spline_path(degree=self._degree)).T
        else:
            return np.array(self._path_smoother.approximate_b_spline_path(degree=self._degree, s=self._smoothing_parameter)).T
        
    def get_social_waypoints(self, granularity=None, interpolate=False):
        """
        Function used to retrieve socially feasible waypoints (from Kivrak et al. 2022)

        :return: array of socially feasible waypoints
        :rtype: numpy array of floats
        """
        path = np.array(self.get_points_path(granularity=granularity, interpolate=interpolate))
        # Empty array of waypoints
        self._social_waypoints = []
        # For every point in the path
        for idx, p in enumerate(path):
            # Starting and ending point must be in the array of socially acceptable waypoints
            if idx == 0 or idx == np.shape(path)[0] - 1:
                self._social_waypoints.append(p)
            else:
                # Compute vector connecting current point and precedent one
                v1 = p[0:2] - path[idx - 1, 0:2]
                # Compute vector connecting subsequent point and current one
                v2 = path[idx + 1, 0:2] - p[0:2]
                # Get angle between them (first clip value of cos(theta) between -1.0 and 1.0)
                theta = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
                # If angle is less then a certain threshold, then add current point to array of waypoints
                if theta < self._theta_threshold:
                    self._social_waypoints.append(p)
        self._path = self._social_waypoints
        return self._social_waypoints
          
    def publish_internal_representation(self):
        self._gridmap_interface.publish_map_internal_representation()

    def set_start(self, p):
        """
        Setter method for start position
        """
        assert_points(p)
        self._start = p

    def set_goal(self, p):
        """
        Setter method for goal position
        """
        assert_points(p)
        self._goal = p

    def get_start(self):
        """
        Getter method for start position
        """
        return self._start
    
    def get_goal(self):
        """
        Getter method for goal position
        """
        return self._goal
    
    def get_mapped_obs_pos(self):
        """
        Function to get the static and mapped obstacle position

        :return: position of mapped obstacles
        :rtype: list[tuple[float]]
        """
        return np.array(self._world.OBS)[:, 0:2]
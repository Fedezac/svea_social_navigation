#! /usr/bin/env python3

import rospy
import numpy as np
from svea_planners.gridmap_interface import GridMapInterface
from svea_planners.path_interface import PathInterface
from svea_planners.astar import AStarPlanner, AStarWorld

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
    THETA_THRESHOLD = 0.11

    def __init__(self, obs_margin=0.05):
        self._obs_margin = obs_margin
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

    def initialize_path_interface(self):
        if self._path is not None:
            self._path_interface = PathInterface(self._path)
        else:
            raise Exception('No path was computed beforehand')
        
    def publish_rviz_path(self):
        self._path_interface.create_pose_path()
        self._path_interface.publish_rviz_path()

    def get_points_path(self, granularity=None):
        if granularity is not None:
            return self._path_interface.get_points_path_reduced(granularity)
        else: 
            return self._path_interface.get_points_path()
        
    def get_social_waypoints(self, granularity=None):
        """
        Function used to retrieve socially feasible waypoints (from Kivrak et al. 2022)

        :return: array of socially feasible waypoints
        :rtype: numpy array of floats
        """
        path = np.array(self.get_points_path(granularity=granularity))
        # Empty array of waypoints
        self._social_waypoints = []
        # For every point in the path
        for idx, p in enumerate(path):
            # Starting and ending point must be in the array of socially acceptable waypoints
            if idx == 0 or idx == np.shape(path)[0] - 1:
                self._social_waypoints.append(p)
            else:
                # Compute vector connecting current point and precedent one
                v1 = p - path[idx - 1]
                # Compute vector connecting subsequent point and current one
                v2 = path[idx + 1] - p
                # Get angle between them
                theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                # If angle is less then a certain threshold, then add current point to array of waypoints
                if theta < self.THETA_THRESHOLD:
                    self._social_waypoints.append(p)
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


"""
==== From now on, main for debugging ====
"""
def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def main():
    rospy.init_node('planner_test')
    debug = load_param('~debug', True)
    
    # TODO: params
    start_x = 7.0
    start_y = 3.6
    goal_x = 2.8
    goal_y = 7.2 
    GRANULARITY = 4
    
    pi = PlannerInterface(obs_margin=0.07)
    pi.set_start([start_x, start_y])
    pi.set_goal([goal_x, goal_y])
    pi.initialize_planner_world()
    pi.compute_path()
    pi.initialize_path_interface()
    path = pi.get_points_path(GRANULARITY)

    if debug:
        pi.publish_internal_representation()

    pi.publish_rviz_path()
    rospy.spin()
    

if __name__ == '__main__':
    main()
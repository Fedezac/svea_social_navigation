#! /usr/bin/env python3
# Plain python imports
from typing import Any
import numpy as np
import rospy

# ROS imports
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class ArtificialPotentialFieldHelper(object):
    """
    Class that given a local costmap computes the APF (only related to obstacles) for given robot 
    """
    def __init__(self, svea_name="svea7", k_a=2, k_r=2, window_len=10):
        """
        Init method for the ArtificialPotentialField class
        """
        self.svea_name = svea_name
        # Get topics names
        self.map_topic = load_param('~local_costmap_topic', '/costmap_node/costmap/costmap')
        self.waypoint_topic = load_param('~waypoint_topic', '/target')

        # Init subscribers
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self._local_costmap_cb, queue_size=1)
        self.waypoint_sub = rospy.Subscriber(self.waypoint_topic, PointStamped, self._waypoint_cb, queue_size=1)

        #  Waypoint variables
        self._waypoint = None

        # Svea states variables
        self._svea_states = None

        # Costmap variables
        self._local_costmap = None
        self._map_width = None
        self._map_height = None
        self._map_x = None
        self._map_y = None
        self._map_resolution = None

        # APF variables
        self.WINDOW_LEN = window_len
        self.k_a = k_a
        self.k_r = k_r
        self.F_a = np.full(self.WINDOW_LEN + 1, -np.inf)
        self.F_r = np.full(self.WINDOW_LEN + 1, -np.inf)

    def _set_states(self, x):
        """
        Callback method for the state subscriber

        :param msg: state message
        :type msg: VehicleState
        """
        self._svea_states = np.array(x)

    def _waypoint_cb(self, msg):
        """
        Callback method for the waypoint subscriber

        :param msg: waypoint message
        :type msg: PointStamped
        """
        self._waypoint = np.array([msg.point.x, msg.point.y])

    def _local_costmap_cb(self, msg):
        """
        Callback method for the local costmap subscriber

        :param msg: map message
        :type msg: OccupancyGrid
        """
        self._map_width = msg.info.width
        self._map_height = msg.info.height
        self._map_resolution = msg.info.resolution
        self._map_x = msg.info.origin.position.x
        self._map_y = msg.info.origin.position.y
        self._local_costmap = np.reshape(msg.data, (self._map_height, self._map_width))
        
    def get_obstacles_position(self):
        """
        Function to retrieve obstacles position, given a local costmap

        :return: position of obstacles
        :rtype: list[tuple[float]]
        """
        if self._local_costmap is not None:
            # Get obstacles indexes
            obs_indexes = np.transpose((self._local_costmap > 70).nonzero())
            obs_positions = np.zeros(np.shape(obs_indexes))
            obs_positions[:, 1] = obs_indexes[:, 0] * self._map_resolution + self._map_y
            obs_positions[:, 0] = obs_indexes[:, 1] * self._map_resolution + self._map_x
            print(f'APF : {obs_positions}')
            return obs_positions
        else:
            return None
        
    def get_map_dimensions(self):
        """
        Function to retrieve the map's dimensions

        :return: maps dimensions
        :rtype: int, int
        """
        if self._map_height is not None and self._map_width is not None:
            return self._map_height, self._map_width
        
    def wait_for_local_costmap(self):
        while self._local_costmap is None:
            rospy.sleep(1)
        
        """
        Function to retrieve the resultant force

        :return: resultant force
        :rtype: float
        """
        return self.F_a + self.F_r
    

if __name__ == '__main__':
    ## Start node ##
    rospy.init_node('apf_node')
    a = ArtificialPotentialFieldHelper(k_a=10, k_r=100, window_len=20)
    a.get_repulsive_forces([[5.60878716, 6.7788254 ],[5.60731558, 6.80588076],[5.60621237, 6.83228813],[5.60550238, 6.85865974],[5.60524131, 6.88509128],[5.6055639 , 6.91207052],[5.60697562, 6.94083662],[5.61051214, 6.97177156],[5.61671512, 7.00309276],[5.62532682, 7.03331632],[5.6359971 , 7.06185886],[5.64805739, 7.0883965 ],[5.66084955, 7.11314632],[5.67406027, 7.13648677],[5.68749305, 7.15861945],[5.70096879, 7.17964629],[5.71433584, 7.19964465],[5.72746704, 7.21868267],[5.74026105, 7.23683041],[5.75264475, 7.25416544],[5.76450291, 7.27066924]])
    rospy.spin()


[2.4        ,2.4,        2.4,        2.60000001, 2.60000001, 2.60000001,  2.80000001, 2.80000001, 5.00000004, 5.00000004, 5.00000004, 5.00000004]        
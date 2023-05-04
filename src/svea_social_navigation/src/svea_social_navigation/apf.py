#! /usr/bin/env python3
# Plain python imports
from typing import Any
import numpy as np
from scipy import spatial
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
    def __init__(self, mapped_obs, svea_name="svea7"):
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

        # Static mapped obstacles
        self._mapped_obs = mapped_obs

        # Costmap variables
        self._local_costmap = None
        self._map_width = None
        self._map_height = None
        self._map_x = None
        self._map_y = None
        self._map_resolution = None

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
            obs_indexes = np.transpose((self._local_costmap > 50).nonzero())
            obs_positions = []
            for idx in obs_indexes:
                pos = [idx[1] * self._map_resolution + self._map_x, idx[0] * self._map_resolution + self._map_y]
                distance, index = spatial.KDTree(self._mapped_obs).query(pos)
                if distance > 1:
                    obs_positions.append(pos)
            return np.array(obs_positions)
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
        

    
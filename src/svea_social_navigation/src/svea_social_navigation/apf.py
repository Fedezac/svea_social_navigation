#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy

# ROS imports
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from svea_msgs.msg import VehicleState as VehicleStateMsg

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class ArtificialPotentialField(object):
    """
    Class that given a local costmap computes the APF (only related to obstacles) for given robot 
    """
    def __init__(self, svea_name="svea7", k_a=2, k_r=2):
        """
        Init method for the ArtificialPotentialField class
        """
        self.svea_name = svea_name
        # Get topics names
        self.map_topic = load_param('~local_costmap_topic', '/costmap_node/costmap/costmap')
        self.waypoint_topic = load_param('~waypoint_topic', '/target')
        self.state_topic = load_param('~state_topic', f'/{self.svea_name}/state')

        # Init subscribers
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self._local_costmap_cb, queue_size=1)
        self.waypoint_sub = rospy.Subscriber(self.waypoint_topic, PointStamped, self._waypoint_cb, queue_size=1)
        self.state_sub = rospy.Subscriber(self.state_topic, VehicleStateMsg, self._state_cb, queue_size=1)

        #  Waypoint variables
        self._waypoint = None

        # Svea state variables
        self._svea_state = None

        # Costmap variables
        self._local_costmap = None
        self._map_width = None
        self._map_height = None
        self._map_x = None
        self._map_y = None
        self._map_resolution = None

        # APF variables
        self.k_a = k_a
        self.k_r = k_r
        self.F_a = 0
        self.F_r = 0

    def _state_cb(self, msg):
        """
        Callback method for the state subscriber

        :param msg: state message
        :type msg: VehicleState
        """
        self._svea_state = np.array([msg.x, msg.y])

    def _waypoint_cb(self, msg):
        """
        Callback method for the waypoint subscriber

        :param msg: waypoint message
        :type msg: PointStamped
        """
        self._waypoint = np.array([msg.point.x, msg.point.y])
        self._compute_attractive_force()

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
        self._compute_repulsive_force()

    def _compute_attractive_force(self):
        """
        Method to compute attractive force
        """
        if self._svea_state is not None:
            self.F_a = 0.5 * self.k_a * np.sqrt((self._svea_state[0] - self._waypoint[0]) ** 2 + (self._svea_state[1] - self._waypoint[1]) ** 2)

    def _compute_repulsive_force(self):
        """
        Method to compute repulsive force
        """
        # TODO: maybe get obstacles starting from 50 of intenties and then scale their repulsive force accordingly
        # TODO: (i.e. less)
        if self._svea_state is not None:
            obs_indexes = np.transpose((self._local_costmap>70).nonzero())
            print(obs_indexes)
            obs_positions = np.zeros(np.shape(obs_indexes))
            obs_positions[:, 1] = obs_indexes[:, 0] * self._map_resolution + self._map_y
            obs_positions[:, 0] = obs_indexes[:, 1] * self._map_resolution + self._map_x
            exp = sum([np.sqrt((self._svea_state[0] - obs[0]) ** 2 + (self._svea_state[1] - obs[1]) ** 2) for obs in obs_positions])
            self.F_r = 0.5 * self.k_r * np.exp(-exp)
            print(self.F_r)

    def get_resultant_force(self):
        return self.F_a + self.F_r

if __name__ == '__main__':
    ## Start node ##
    rospy.init_node('apf_node')
    a = ArtificialPotentialField()
    rospy.spin()

        
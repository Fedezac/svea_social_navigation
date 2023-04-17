#!/usr/bin/env python

"""
Module containing localization interface for motion capture
"""

from __future__ import division
from threading import Thread, Event
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from svea.states import VehicleState
from tf.transformations import quaternion_matrix, euler_from_matrix, euler_matrix, quaternion_from_euler
import numpy as np
import math

__license__ = "MIT"
__maintainer__ = "Frank Jiang"
__email__ = "frankji@kth.se "
__status__ = "Development"


class MotionCaptureInterface(object):
    """Interface handling the reception of state information from the
    motion capture system. This object can take on several callback
    functions and execute them as soon as state information is
    available.

    :param mocap_name: Name of mocap model in Qualisys software;
                                The name will be effectively be added as a
                                namespace to the topics used by the
                                corresponding localization node i.e
                                `qualisys/model_name/odom`, defaults to
                                ''
    :type mocap_name: str, optional
    """

    def __init__(self, mocap_name=''):
        self.model_name = mocap_name
        self._odom_sub = None
        self._vel_sub = None

        self._curr_vel_twist = None
        self.state = VehicleState()
        self.last_time = float('nan')

        self._x_offset = 0.0 # [m]
        self._y_offset = 0.0

        self.is_ready = False
        self._ready_event = Event()
        rospy.on_shutdown(self._shutdown_callback)

        # Offset angle between the mocap frame (of the real world) and the map frame
        self.OFFSET_ANGLE = -math.pi/2
        self.T_MATRIX_4 = euler_matrix(0, 0, self.OFFSET_ANGLE)
        # Create rotation matrix given the offset angle and linear misalignment between mocap and map frames
        self.T_MATRIX_4[0:3,3] = np.transpose(np.array([5.619764999999999, 5.870124000000001, 0]))

        # list of functions to call whenever a new state comes in
        self.callbacks = []

    def update_name(self, name):
        self.model_name = name
        self._odom_topic = 'qualisys/' + self.model_name + '/odom'
        self._vel_topic = 'qualisys/' + self.model_name + '/velocity'
        # check if old subs need to be removed
        if not self._odom_sub is None:
            self._odom_sub.unregister()
        if not self._vel_sub is None:
            self._vel_sub.unregister()
        self._start_listen()

    def set_model_offset(self, x, y):
        self._x_offset = x
        self._y_offset = y

    def start(self):
        """Spins up ROS background thread; must be called to start
        receiving data

        :return: itself
        :rtype: MotionCaptureInterface
        """
        Thread(target=self._init_and_spin_ros, args=()).start()
        return self

    def _wait_until_ready(self, timeout=20.0):
        tic = rospy.get_time()
        self._ready_event.wait(timeout)
        toc = rospy.get_time()
        wait = toc - tic
        return wait < timeout

    def _shutdown_callback(self):
        self._ready_event.set()

    def _init_and_spin_ros(self):
        rospy.loginfo("Starting Motion Capture Interface Node for "
                      + self.model_name)
        self.node_name = 'motion_capture_node'
        self.update_name(self.model_name)
        self.is_ready = self._wait_until_ready()
        if not self.is_ready:
            rospy.logwarn("Motion Capture not responding during start of "
                          "Motion Caputer. Setting ready anyway.")
        self.is_ready = True
        rospy.loginfo("{} Motion Capture Interface successfully initialized"
                      .format(self.model_name))

        rospy.spin()

    def _start_listen(self):
        self._odom_sub = rospy.Subscriber(self._odom_topic,
                                           Odometry,
                                           self._read_odom_msg,
                                           tcp_nodelay=True,
                                           queue_size=1)
        self._vel_sub = rospy.Subscriber(self._vel_topic,
                                        TwistStamped,
                                        self._read_vel_msg,
                                        tcp_nodelay=True,
                                        queue_size=1)

    def fix_twist(self, odom_msg):
        odom_msg.twist.twist = self._curr_vel_twist
        return odom_msg
    
    def _correct_mocap_coordinates(self, msg):
        """
        Method used to correct the mocap pose (if some misalignment between its frame and the map frame is present)
        
        :param x: x coordinate to be corrected 
        :type x: float
        :param y: y coordinate to be corrected 
        :type y: float
        :param quaternion: quaternion used to extract and correct yaw angle 
        :type quaternion: Quaternion

        :return: rotate_point[0] corrected x coordinate
        :rtype: float
        :return: rotate_point[1] corrected y coordinate
        :rtype: float
        :return: mocap_yaw corrected yaw angle
        :rtype: float
        """
        # Get svea's rotation matrix from pose quaternion
        svea_T_mocap = quaternion_matrix([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        # Add translational part to transofmration matrix
        svea_T_mocap[0:3,3] = np.transpose(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, 0]))

        # Apply 4 dimension square rotation matrix (rotate svea's yaw)
        svea_T_map = np.matmul(self.T_MATRIX_4, svea_T_mocap)

        # Get correct yaw (from manipulated rotation matrix)
        (mocap_roll, mocap_pitch, mocap_yaw) = euler_from_matrix(svea_T_map)
        
        msg.pose.pose.position.x = svea_T_map[0, 3]
        msg.pose.pose.position.y = svea_T_map[1, 3]
        quat = quaternion_from_euler(mocap_roll, mocap_pitch, mocap_yaw)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        return msg
    
    def _compute_vehicle_velocity(self, msg):
        """
        Method used to compute the vehicle's velocity given the mocap's twist
        
        :param quaternion: quaternion used to extract and correct velocity
        :type quaternion: Quaternion

        :return: v[0] vehicle velocity
        :rtype: float
        """

        # Apply 4 dimension square rotation matrix (rotate svea's yaw)
        corr_linear_twist = np.matmul(self.T_MATRIX_4[0:3,0:3], np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]).T)
        corr_angular_twist = np.matmul(self.T_MATRIX_4[0:3,0:3], np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]).T)
        
        msg.twist.twist.linear.x = corr_linear_twist[0]
        msg.twist.twist.linear.y = corr_linear_twist[1]
        msg.twist.twist.linear.z = corr_linear_twist[2]

        msg.twist.twist.angular.x = corr_angular_twist[0]
        msg.twist.twist.angular.y = corr_angular_twist[1]
        msg.twist.twist.angular.z = corr_angular_twist[2]
        
        return msg

    def _read_odom_msg(self, msg):
        if not self._curr_vel_twist is None:
            msg = self.fix_twist(msg)
            msg = self._correct_mocap_coordinates(msg)
            msg = self._compute_vehicle_velocity(msg)
            self.state.odometry_msg = msg
            # apply model offsets (if any)
            self.state.x += self._x_offset
            self.state.y += self._y_offset
            self.last_time = rospy.get_time()
            self._ready_event.set()
            self._ready_event.clear()

            for cb in self.callbacks:
                cb(self.state)

    def _read_vel_msg(self, msg):
        self._curr_vel_twist = msg.twist

    def add_callback(self, cb):
        """Add state callback. Every function passed into this method
        will be called whenever new state information comes in from the
        motion capture system.

        :param cb: A callback function intended for responding to the
                   reception of state info
        :type cb: function
        """
        self.callbacks.append(cb)

    def remove_callback(self, cb):
        """Remove callback so it will no longer be called when state
        information is received

        :param cb: A callback function that should be no longer used
                   in response to the reception of state info
        :type cb: function
        """
        while cb in self.callbacks:
            self.callbacks.pop(self.callbacks.index(cb))

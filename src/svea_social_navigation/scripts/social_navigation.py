#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy
from copy import deepcopy

# SVEA imports
from svea.controllers.mpc import MPC
from svea.sensors import Lidar
from svea.models.bicycle_mpc import BicycleModel
from svea.models.bicycle import SimpleBicycleModel
from svea.states import VehicleState
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import LocalizationInterface, ActuationInterface
from svea_mocap.mocap import MotionCaptureInterface
from svea.data import TrajDataHandler, RVIZPathHandler
from svea_planners.planner_interface import PlannerInterface 

# ROS imports
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Path

def load_param(name, value=None):
    """
    Function used to get parameters from ROS parameter server

    :param name: name of the parameter
    :type name: string
    :param value: default value of the parameter, defaults to None
    :type value: _type_, optional
    :return: value of the parameter
    :rtype: _type_
    """
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


def assert_points(pts):
    """
    Method used to assert that points are well formed

    :param pts: array of points
    :type pts: list(2-ple(float))
    """
    assert isinstance(pts, (list, tuple)), 'points is of wrong type, expected list'
    for xy in pts:
        assert isinstance(xy, (list, tuple)), 'points contain an element of wrong type, expected list of two values (x, y)'
        assert len(xy), 'points contain an element of wrong type, expected list of two values (x, y)'
        x, y = xy
        assert isinstance(x, (int, float)), 'points contain a coordinate pair wherein one value is not a number'
        assert isinstance(y, (int, float)), 'points contain a coordinate pair wherein one value is not a number'

def publish_initialpose(state, n=10):
    """
    Method for publishing initial pose

    :param state: vehicle state
    :type state: VehicleState
    :param n: _description_, defaults to 10
    :type n: int, optional
    """
    p = PoseWithCovarianceStamped()
    p.header.frame_id = 'map'
    p.pose.pose.position.x = state.x
    p.pose.pose.position.y = state.y

    q = quaternion_from_euler(0, 0, state.yaw)
    p.pose.pose.orientation.z = q[2]
    p.pose.pose.orientation.w = q[3]

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rate = rospy.Rate(10)

    for _ in range(n):
        pub.publish(p)
        rate.sleep()

def lists_to_pose_stampeds(x_list, y_list, yaw_list=None, t_list=None):
    poses = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]

        curr_pose = PoseStamped()
        curr_pose.header.frame_id = 'map'
        curr_pose.pose.position.x = x
        curr_pose.pose.position.y = y

        if not yaw_list is None:
            yaw = yaw_list[i]
            quat = quaternion_from_euler(0.0, 0.0, yaw)
            curr_pose.pose.orientation.x = quat[0]
            curr_pose.pose.orientation.y = quat[1]
            curr_pose.pose.orientation.z = quat[2]
            curr_pose.pose.orientation.w = quat[3]

        if not t_list is None:
            t = t_list[i]
            curr_pose.header.stamp = rospy.Time(secs=t)
        else:
            curr_pose.header.stamp = rospy.Time.now()

        poses.append(curr_pose)
    return poses

def find_nearest(array, value):   
    idx = (np.abs(array - value)).argmin();     
    return idx

class SocialNavigation(object):
    DELTA_TIME = 0.1
    GOAL_THRESH = 0.1
    TARGET_VELOCITY = 0.3
    WINDOW_LEN = 1
    MAX_WAIT = 1.0/10.0 # no slower than 10Hz
    POINTS = []

    def __init__(self):
        """
        Init method for SocialNavigation class
        """
        rospy.init_node('svea_social_navigation')

        # Get parameters
        self.IS_SIM = load_param('~is_sim', False)
        self.USE_RVIZ = load_param('~use_rviz', True)
        self.REMOTE_RVIZ = load_param('~remote_rviz', False)
        self.IS_MOCAP = load_param('~is_mocap', True)
        self.STATE = load_param('~state', [0, 0, 0, 0])
        self.GOAL = load_param('~goal', [0, 0])

        # Define publisher for MPC predicted path
        self.pred_path_pub = rospy.Publisher("pred_path", Path, queue_size=1, latch=True)

        # Initilize vehicle state
        self.state = VehicleState(*self.STATE)
        self.last_state_time = None
        # Publish initial pose
        publish_initialpose(self.state)

        # Instatiate RVIZPathHandler object if publishing to RVIZ
        if self.IS_SIM or self.REMOTE_RVIZ:
            self.data_handler = RVIZPathHandler()

        if self.IS_SIM:
            # Simulator needs a model to simulate
            self.sim_model = SimpleBicycleModel(self.state)

            # Start the simulator immediately, but paused
            self.simulator = SimSVEA(self.sim_model,
                                     dt=self.DELTA_TIME,
                                     run_lidar=True,
                                     start_paused=True).start()

        # Create vehicle model object
        self.model = BicycleModel(initial_state=self.state, dt=self.DELTA_TIME)

        # Create MPC controller object
        self.controller = MPC(
            self.model,
            N=self.WINDOW_LEN,
            Q=[5, 5, 50, 7],
            R=[1, 2],
            x_lb=[-100, -100, -0.5, -2*np.pi],
            x_ub=[100, 100, 0.6, 2*np.inf],
            u_lb=[-1, -np.pi / 5],
            u_ub=[1.5, np.pi / 5],
        )

        # Start actuation interface 
        self.actuation = ActuationInterface().start()
        # Start localization interface based on which localization method is being used
        if self.IS_MOCAP:
            self.localizer = MotionCaptureInterface().start()
        else:
            self.localizer = LocalizationInterface().start()
        # Start lidar
        self.lidar = Lidar().start()
        # Start simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

    def wait_for_state_from_localizer(self):
        """Wait for a new state to arrive, or until a maximum time
        has passed since the last state arrived.

        :return: New state when it arrvies, if it arrives before max
                 waiting time, otherwise None
        :rtype: VehicleState, or None
        """
        time = rospy.get_time()
        if self.last_state_time is None:
            timeout = None
        else:
            timeout = self.MAX_WAIT - (time - self.last_state_time)
        if timeout is None or timeout <= 0:
            return deepcopy(self.state)

        self.localizer.ready_event.wait(timeout)
        wait = rospy.get_time() - time
        if wait < self.MAX_WAIT:
            return deepcopy(self.state)
        else:
            return None

    def plan(self):
        debug = False
        
        pi = PlannerInterface()
        pi.set_start([self.state.x, self.state.y])
        pi.set_goal([self.GOAL[0], self.GOAL[1]])
        pi.initialize_planner_world()
        pi.compute_path()
        pi.initialize_path_interface()
        # Get path 
        self.path = pi.get_points_path()
        # Convert it to np array
        self.path = np.array(self.path)
        print(self.path)

        # If debug mode is on, publish map's representation on RVIZ
        if debug:
            pi.publish_internal_representation()
        # Publish global path on rviz
        pi.publish_rviz_path()

        # Create path structure for MPC
        self.path = np.hstack((self.path, np.full((np.shape(self.path)[0], 1), self.TARGET_VELOCITY)))
        self.path = np.hstack((self.path, np.zeros((np.shape(self.path)[0], 1))))

    def keep_alive(self):
        # TODO: condition for stopping should be to be on the goal
        return not (rospy.is_shutdown())

    def run(self):
        """
        Run method
        """
        # Plan a feasible path
        self.plan()
        self.waypoint_idx = 1
        # Spin until alive and if localizer is reasy
        while self.keep_alive():
            self.spin()

    def spin(self):
        safe = self.localizer.is_ready
        if self.IS_SIM:
            self.state = self.sim_model.state
        else:
            # Wait for state
            self.state = self.wait_for_state_from_localizer()
        print(f'State: {self.state.x, self.state.y, self.state.v, self.state.yaw}')
        # Create new initial state for MPC
        current_state = np.vstack([self.state.x, self.state.y, self.state.v, self.state.yaw])
        # TODO: feasible way of getting index of next waypoint
        if np.linalg.norm(current_state[0:2] - self.path[0:2, self.waypoint_idx]) < self.GOAL_THRESH:
            self.waypoint_idx += 1
        print(f'Closest state: {self.path[self.waypoint_idx,:]}')
        if self.waypoint_idx + self.WINDOW_LEN + 1 >= np.shape(self.path)[0]:
            # TODO: safe way to have fake N points when getting closer to the end of the path 
            last_iteration_points = self.path[self.waypoint_idx:, :]
            while np.shape(last_iteration_points)[0] < self.WINDOW_LEN + 1:
                last_iteration_points = np.vstack((last_iteration_points, self.path[-1, :]))
            print(last_iteration_points)
            u, predicted_state = self.controller.get_ctrl(current_state, last_iteration_points[:, :].T)
        else:
            u, predicted_state = self.controller.get_ctrl(current_state, self.path[self.waypoint_idx:self.waypoint_idx + self.WINDOW_LEN + 1, :].T)

        # Get optimal velocity and steering controls
        velocity = u[0, 0]
        steering = u[1, 0]
        # Send control to actuator interface
        if safe:
            self.actuation.send_control(steering, velocity)

        # If model is simulated, then update new state
        if self.IS_SIM:
            self.sim_model.update(steering, velocity, self.DELTA_TIME)
            
        current_state = np.vstack([self.state.x, self.state.y, self.state.v, self.state.yaw])
        if np.linalg.norm(current_state[0:2] - self.path[0:2, self.waypoint_idx]) < self.GOAL_THRESH:
            self.waypoint_idx += 1

        # Get MPC predicted path as planned path
        x_pred = predicted_state[0, :]
        y_pred = predicted_state[1, :]
        # Visualize predicted local tracectory
        new_pred = lists_to_pose_stampeds(list(x_pred), list(y_pred))
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path.poses = new_pred
        self.pred_path_pub.publish(path)

        # Visualize data
        self.data_handler.visualize_data()
        self.data_handler.update_target((self.path[self.waypoint_idx, 0], self.path[self.waypoint_idx, 1]))

if __name__ == '__main__':
    ## Start node ##
    SocialNavigation().run()
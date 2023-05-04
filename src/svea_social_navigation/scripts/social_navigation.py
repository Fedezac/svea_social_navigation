#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy
from copy import deepcopy

from matplotlib import pyplot as plt

# SVEA imports
from svea.controllers.mpc import MPC
from svea.sensors import Lidar
from svea.models.bicycle_mpc import BicycleModel
from svea.models.bicycle import SimpleBicycleModel
from svea.states import VehicleState
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import LocalizationInterface, ActuationInterface
from svea_mocap.mocap import MotionCaptureInterface
from svea.data import RVIZPathHandler
from svea_planners.planner_interface import PlannerInterface
from svea_social_navigation.apf import ArtificialPotentialFieldHelper

# ROS imports
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
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

class SocialNavigation(object):
    DELTA_TIME = 0.1
    GOAL_THRESH = 0.2
    STRAIGHT_SPEED = 0.3
    TURN_SPEED = 0.2
    WINDOW_LEN = 20
    MAX_WAIT = 1.0/10.0 # no slower than 10Hz

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
        self.SVEA_NAME = load_param('~svea_name', 'svea7')
        self.DYNAMIC_OBSTACLE_TOPIC = load_param('~dynamic_obstacle_topic', '/dynamic_obstacle')
        # Define publisher for MPC predicted path
        self.pred_path_pub = rospy.Publisher("pred_path", Path, queue_size=1, latch=True)

        # Initialize vehicle state
        self.state = VehicleState(*self.STATE)
        self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        self.last_state_time = None
        # Publish initial pose
        publish_initialpose(self.state)

        # Instatiate RVIZPathHandler object if publishing to RVIZ
        self.data_handler = RVIZPathHandler()

        # Define planner interface
        self.pi = PlannerInterface(obs_margin=0.2, degree=3, s=None, theta_threshold=0.3)
        # Set start and goal point
        self.pi.set_start([self.state.x, self.state.y])
        self.pi.set_goal([self.GOAL[0], self.GOAL[1]])
        # Initialize planner world
        self.pi.initialize_planner_world()

        if self.IS_SIM:
            self.dynamic_obs_pos = [-100000.0, -100000.0]
            # Subscriber to dynamic obstacle topic
            self.dynamic_obstacle_sub = rospy.Subscriber(self.DYNAMIC_OBSTACLE_TOPIC, PointStamped, self._dynamic_obstacle_cb, queue_size=1)
            # Simulator needs a model to simulate
            self.sim_model = SimpleBicycleModel(self.state)
            # Start the simulator immediately, but paused
            self.simulator = SimSVEA(self.sim_model,
                                     vehicle_name=self.SVEA_NAME,
                                     dt=self.DELTA_TIME,
                                     run_lidar=True,
                                     start_paused=True).start()
        else:
            # Start lidar
            self.lidar = Lidar().start()
            # Start actuation interface 
            self.actuation = ActuationInterface().start()
            # Start localization interface based on which localization method is being used
            if self.IS_MOCAP:
                self.localizer = MotionCaptureInterface(self.SVEA_NAME).start()
            else:
                self.localizer = LocalizationInterface().start()

        # Start simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

        # Create APF object
        self.apf = ArtificialPotentialFieldHelper(svea_name=self.SVEA_NAME, mapped_obs=self.pi.get_mapped_obs_pos())
        self.apf.wait_for_local_costmap()
        # Create vehicle model object
        self.model = BicycleModel(initial_state=self.x0, dt=self.DELTA_TIME)
        # Define variable bounds
        x_b = np.array([np.inf, np.inf, 0.5, np.inf])
        u_b = np.array([1.7, np.deg2rad(40)])
        # Create MPC controller object
        self.controller = MPC(
            self.model,
            N=self.WINDOW_LEN,
            Q=[7, 7, .1, .1],
            R=[1, 1],
            S=[5],
            x_lb=-x_b,
            x_ub=x_b,
            u_lb=-u_b,
            u_ub=u_b,
            n_obstacles=self.apf.get_map_dimensions()[0] * self.apf.get_map_dimensions()[1],
            verbose=False
        )
        # Create matrix of local obstacles
        self.local_obstacles = np.full((2, self.apf.get_map_dimensions()[0] * self.apf.get_map_dimensions()[1]), np.array([[-100000.0, -100000.0]]).T)

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

    def _dynamic_obstacle_cb(self, msg):
        self.dynamic_obs_pos = [msg.point.x, msg.point.y]

    def plan(self):
        debug = False
        # Compute safe global path
        self.pi.compute_path()
        # Init visualize path interface
        self.pi.initialize_path_interface()
        # Get path 
        #b_spline_path = np.array(self.pi.get_points_path(interpolate=True))
        # Get smoothed path and extract social waypoints (other possible nice combination of parameters for path
        # smoothing is interpolate=False and degree=4)
        b_spline_path = np.array(self.pi.get_social_waypoints(interpolate=True))
        # Create array for MPC reference
        self.path = np.zeros(np.shape(b_spline_path))
        self.path[:, 0] = b_spline_path[:, 0]
        self.path[:, 1] = b_spline_path[:, 1]
        self.path[:, 2] = [self.STRAIGHT_SPEED if abs(curv) < 1e-2 else self.TURN_SPEED for curv in b_spline_path[:, 3]]
        self.path[:, 3] = b_spline_path[:, 2]
        # Re-initialize path interface to visualize on RVIZ socially aware path
        self.pi.initialize_path_interface()
        print(f'Social navigation path: {self.path[:, 0:2]} size, {np.shape(self.path)[0]}')

        # If debug mode is on, publish map's representation on RVIZ
        if debug:
            self.pi.publish_internal_representation()
        # Publish global path on rviz
        self.pi.publish_rviz_path()
            
    def _visualize_data(self, x_pred, y_pred, velocity, steering):
        # Visualize predicted local tracectory
        new_pred = lists_to_pose_stampeds(list(x_pred), list(y_pred))
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path.poses = new_pred
        self.pred_path_pub.publish(path)

        # Visualize data
        if self.IS_SIM:
            self.data_handler.log_state(self.sim_model.state)
        else:
            self.data_handler.log_state(self.state)
        self.data_handler.log_ctrl(steering, velocity, rospy.get_time())
        self.data_handler.update_target((self.path[self.waypoint_idx, 0], self.path[self.waypoint_idx, 1]))
        self.data_handler.visualize_data()

    def keep_alive(self):
        """
        Keep alive function based on the distance to the goal and current state of node

        :return: True if the node is still running, False otherwise
        :rtype: boolean
        """
        distance = np.linalg.norm(np.array(self.GOAL) - np.array([self.x0[0], self.x0[1]]))
        return not (rospy.is_shutdown() or distance < self.GOAL_THRESH)

    def run(self):
        """
        Run method
        """
        # Plan a feasible path
        self.plan()
        self.waypoint_idx = 0
        # Spin until alive
        while self.keep_alive():
            self.spin()
            #rospy.sleep(0.1)
        print('--- GOAL REACHED ---')

    def spin(self):
        """
        Main method
        """
        # Get svea state
        if not self.IS_SIM:
            safe = self.localizer.is_ready
            # Wait for state from localization interface
            self.state = self.wait_for_state_from_localizer()
            self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        else:
            self.x0 = [self.sim_model.state.x, self.sim_model.state.y, self.sim_model.state.v, self.sim_model.state.yaw]
            print(f'State: {self.x0}')

        # Fill obstacle array with own position (so that repulsive force is 0)
        self.local_obstacles = np.full((2, self.apf.get_map_dimensions()[0] * self.apf.get_map_dimensions()[1]), np.array([[-100000.0, -100000.0]]).T)

        # Get position of obstacles deteceted in the local costmap 
        obs = np.array(self.apf.get_obstacles_position()).T
        # If obstacles have been detected, insert them into the array
        if len(obs) > 0:
            self.local_obstacles[:, 0:np.shape(obs)[1]] = obs
            self.local_obstacles[:, np.shape(obs)[1]] = self.dynamic_obs_pos
        else:
            self.local_obstacles[:, 0] = self.dynamic_obs_pos
        closest_obs = np.linalg.norm(np.array([self.x0[0:2]]).T - self.local_obstacles, axis=0).argmin()

        plt.clf()
        plt.scatter(np.array(self.pi._world.OBS)[:, 0], np.array(self.pi._world.OBS)[:, 1])
        plt.scatter(self.x0[0], self.x0[1])
        if len(obs)>0:
            plt.scatter(obs[0, :], obs[1, :])
        plt.scatter(self.local_obstacles[0, closest_obs], self.local_obstacles[1, closest_obs])
        plt.draw()
        plt.pause(0.01)
        
        # Get next waypoint index (by computing offset between robot and each point of the path), wrapping it in case of
        # index out of bounds
        self.waypoint_idx = np.minimum(np.argmin(np.linalg.norm(self.path[:, 0:2] - np.array([self.x0[0], self.x0[1]]), axis=1)) + 1, np.shape(self.path)[0] - 1)

        # If there are not enough waypoints for concluding the path, then fill in the waypoints array with the desiderd
        # final goal
        if self.waypoint_idx + self.WINDOW_LEN + 1 >= np.shape(self.path)[0]:
            last_iteration_points = self.path[self.waypoint_idx:, :]
            while np.shape(last_iteration_points)[0] < self.WINDOW_LEN + 1:
                last_iteration_points = np.vstack((last_iteration_points, self.path[-1, :]))
            u, predicted_state = self.controller.get_ctrl(self.x0, last_iteration_points[:, :].T, self.local_obstacles)
        else:
            u, predicted_state = self.controller.get_ctrl(self.x0, self.path[self.waypoint_idx:self.waypoint_idx + self.WINDOW_LEN + 1, :].T, self.local_obstacles)
        #u, predicted_state = self.controller.get_ctrl(self.x0, self.path[self.waypoint_idx, :], self.local_obstacles)

        # Get optimal velocity (by integrating once the acceleration command and summing it to the current speed) and steering controls
        velocity = u[0, 0] * self.DELTA_TIME + self.x0[2]
        steering = u[1, 0]
        print(f'Optimal control (acceleration, steering): {u[0, 0], steering}')
        
        # Send control to actuator interface
        if not self.IS_SIM and safe:
            self.actuation.send_control(steering, velocity)

        # If model is simulated, then update new state
        if self.IS_SIM:
            self.sim_model.update(steering, velocity, self.DELTA_TIME)
            
        # Visualize data on RVIZ
        self._visualize_data(predicted_state[0, :], predicted_state[1, :], velocity, steering)


if __name__ == '__main__':
    ## Start node ##
    SocialNavigation().run()
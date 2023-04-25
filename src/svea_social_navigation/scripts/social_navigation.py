#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy
from matplotlib import pyplot as plt
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
from svea.data import RVIZPathHandler
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
    GOAL_THRESH = 0.1
    TARGET_VELOCITY = 0.2
    WINDOW_LEN = 10
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
        # Define publisher for MPC predicted path
        self.pred_path_pub = rospy.Publisher("pred_path", Path, queue_size=1, latch=True)

        # Initilize vehicle state
        self.state = VehicleState(*self.STATE)
        self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        self.last_state_time = None
        # Publish initial pose
        publish_initialpose(self.state)

        # Create vehicle model object
        self.model = BicycleModel(initial_state=self.x0, dt=self.DELTA_TIME)
        # Create MPC controller object
        self.controller = MPC(
            self.model,
            N=self.WINDOW_LEN,
            Q=[5, 5, 50, 7],
            R=[1, 2],
            x_lb=[-100, -100, -0.5, -2*np.pi],
            x_ub=[100, 100, 0.6, 2*np.inf],
            u_lb=[-1, -np.deg2rad(40)],
            u_ub=[1.5, np.deg2rad(40)],
            verbose=False
        )

        # Instatiate RVIZPathHandler object if publishing to RVIZ
        self.data_handler = RVIZPathHandler()

        if self.IS_SIM:
            # Simulator needs a model to simulate
            self.sim_model = SimpleBicycleModel(self.state)
            # TODO: decomment when a solution will be found
            # Start the simulator immediately, but paused
            #self.simulator = SimSVEA(self.sim_model,
            #                         vehicle_name=self.SVEA_NAME,
            #                         dt=self.DELTA_TIME,
            #                         run_lidar=True,
            #                         start_paused=True).start()
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
        #if self.IS_SIM:
        #    self.simulator.toggle_pause_simulation()

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
        #self.path = np.array(pi.get_points_path())
        self.path = np.array(pi.get_social_waypoints())
        print(f'Social navigation path: {self.path} size, {np.shape(self.path)[0]}')

        # If debug mode is on, publish map's representation on RVIZ
        if debug:
            pi.publish_internal_representation()
        # Publish global path on rviz
        pi.publish_rviz_path()
        # Create path structure for MPC
        self.path = np.hstack((self.path, np.full((np.shape(self.path)[0], 1), self.TARGET_VELOCITY)))
        self.path = np.hstack((self.path, np.zeros((np.shape(self.path)[0], 1))))
        for i, p in enumerate(self.path):
            if i != np.shape(self.path)[0] - 1:
                self.path[i, 3] = np.arctan2(self.path[i + 1, 1] - self.path[i, 1], self.path[i + 1, 0] - self.path[i, 0])
            else:
                self.path[i, 3] = self.path[i - 1, 3]
        
            
    def _visualize_data(self, x_pred, y_pred, velocity, steering):
        # Visualize predicted local tracectory
        new_pred = lists_to_pose_stampeds(list(x_pred), list(y_pred))
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path.poses = new_pred
        self.pred_path_pub.publish(path)

        # Visualize data
        self.data_handler.log_state(self.state)
        self.data_handler.log_ctrl(steering, velocity, rospy.get_time())
        self.data_handler.update_target((self.path[self.waypoint_idx, 0], self.path[self.waypoint_idx, 1]))
        self.data_handler.visualize_data()

    def keep_alive(self):
        distance = np.linalg.norm(np.array(self.GOAL) - np.array([self.state.x, self.state.y]))
        return not (rospy.is_shutdown() or distance < self.GOAL_THRESH)

    def run(self):
        """
        Run method
        """
        # Plan a feasible path
        self.plan()
        self.waypoint_idx = 0
        self.traj = []
        print(self.x0)
        # Spin until alive
        while self.keep_alive():
            self.spin()
        print('ENDING')

    def spin(self):
        if not self.IS_SIM:
            safe = self.localizer.is_ready
            # Wait for state from localization interface
            self.state = self.wait_for_state_from_localizer()
            self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        else:
            self.x0 = [self.sim_model.state.x, self.sim_model.state.y, self.sim_model.state.v, self.sim_model.state.yaw]
            print(f'State: {self.x0}')

        # TODO: use lateral offset from path to get next waypoint
        while np.linalg.norm(self.x0[0:2] - self.path[self.waypoint_idx, 0:2]) < self.GOAL_THRESH and self.waypoint_idx < np.shape(self.path)[0] - 1:
            print(f'SWITCHING TO NEXT WAYPOINT {self.path[self.waypoint_idx, :]}')
            print()
            self.waypoint_idx += 1

        # If there are not enough waypoints for concluding the path, then fill in the waypoints array with the desiderd
        # final goal
        if self.waypoint_idx + self.WINDOW_LEN + 1 >= np.shape(self.path)[0]:
            # TODO: safe way to have fake N points when getting closer to the end of the path 
            last_iteration_points = self.path[self.waypoint_idx:, :]
            while np.shape(last_iteration_points)[0] < self.WINDOW_LEN + 1:
                last_iteration_points = np.vstack((last_iteration_points, self.path[-1, :]))
            u, predicted_state = self.controller.get_ctrl(self.x0, last_iteration_points[:, :].T)
        else:
            u, predicted_state = self.controller.get_ctrl(self.x0, self.path[self.waypoint_idx:self.waypoint_idx + self.WINDOW_LEN + 1, :].T)

        # Get optimal velocity and steering controls
        velocity = u[0, 0]
        steering = u[1, 0]
        print(f'Optimal control: {velocity, steering}')
        
        # Send control to actuator interface
        if not self.IS_SIM and safe:
            self.actuation.send_control(steering, velocity)

        # If model is simulated, then update new state
        if self.IS_SIM:
            self.sim_model.update(steering, velocity, self.DELTA_TIME)

        self.traj.append([self.x0[0], self.x0[1]])
        plt.clf()
        plt.plot(np.asarray(self.path)[:, 0], np.asarray(self.path)[:, 1], 'bo', alpha=0.3)
        plt.plot(np.asarray(self.path)[self.waypoint_idx, 0], np.asarray(self.path)[self.waypoint_idx, 1], 'go', alpha=0.5)
        plt.plot(np.asarray(self.traj)[:, 0], np.asarray(self.traj)[:, 1], 'r-', alpha=0.7)
        plt.plot(np.asarray(predicted_state).T[:, 0], np.asarray(predicted_state).T[:, 1], 'y-')
        plt.draw()
        plt.pause(0.01)
            
        # Visualize data on RVIZ
        self._visualize_data(predicted_state[0, :], predicted_state[1, :], velocity, steering)


if __name__ == '__main__':
    ## Start node ##
    SocialNavigation().run()
#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy
from copy import deepcopy, copy

# SVEA imports
from svea.controllers.social_mpc import SMPC
from svea.sensors import Lidar
from svea.models.bicycle_mpc import BicycleModel
from svea.models.bicycle import SimpleBicycleModel
from svea.states import VehicleState
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import LocalizationInterface, ActuationInterface
from svea_mocap.mocap import MotionCaptureInterface
from svea.data import RVIZPathHandler
from svea_planners.astar import AStarPlanner, AStarWorld
from svea_planners.planner_interface import PlannerInterface
from svea_planners.bspline_smoother import BSpline
from svea_social_navigation.apf import ArtificialPotentialFieldHelper
from svea_social_navigation.static_unmapped_obstacle_simulator import StaticUnmappedObstacleSimulator
from svea_social_navigation.dynamic_obstacle_simulator import DynamicObstacleSimulator
from svea_social_navigation.sfm_helper import SFMHelper
from svea_social_navigation.measurement_node import SocialMeasurement

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

def increasing_sigmoid(xmin, xmax, ymin, ymax, x):
    """
    Decreasing sigmoid function for weight tuning

    :param xmin: minimum x value
    :type xmin: float
    :param xmax: maximum x value 
    :type xmax: float
    :param ymin: minimum y value
    :type ymin: float
    :param ymax: maximum y value
    :type ymax: float
    :param x: input 
    :type x: float
    :return: f(x)
    :rtype: float
    """
    if x <= xmin:
        y = ymin
    elif x >= xmax:
        y = ymax
    else:
        cosarg = (x - xmin) * np.pi / (xmax - xmin) + np.pi
        y = (ymax - ymin) * (0.5 * np.cos(cosarg) + 0.5) + ymin
    return y

def decreasing_sigmoid(xmin, xmax, ymin, ymax, x):
    """
    Decreasing sigmoid function for weight tuning

    :param xmin: minimum x value
    :type xmin: float
    :param xmax: maximum x value 
    :type xmax: float
    :param ymin: minimum y value
    :type ymin: float
    :param ymax: maximum y value
    :type ymax: float
    :param x: input 
    :type x: float
    :return: f(x)
    :rtype: float
    """
    if x <= xmin:
        y = ymax
    elif x >= xmax:
        y = ymin
    else:
        cosarg = (x - xmin) * np.pi / (xmax - xmin)
        y = (ymax - ymin) * (0.5 * np.cos(cosarg) + 0.5) + ymin
    return y

class SocialNavigation(object):
    WINDOW_LEN = 10
    DELTA_TIME = 0.1
    DELTA_TIME_REAL = 0.3
    GOAL_THRESH = 0.2
    STRAIGHT_SPEED = 0.7
    TURN_SPEED = 0.5
    MAX_N_STATIC_OBSTACLES = 10
    MAX_N_DYNAMIC_OBSTACLES = 10
    MAX_N_PEDESTRIANS = 10
    MAX_WAIT = 1.0/10.0 # no slower than 10Hz
    # Parameter for sigmoid weight tuning
    X_MIN = 0.6
    X_MAX = 2.85
    Y_MIN_SFM = 0.5
    Y_MAX_SFM = 1.5
    Y_MIN_APF = 0.5
    Y_MAX_APF = 1.5
    Y_MIN_TRAJ = 0.5
    Y_MAX_TRAJ = 1.5
    # Weights 
    Q = np.array([20.0, 20.0, 50.0, .1])
    S = np.array([120.0, 150.0, 200.0])

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
        self.SVEA_NAME = load_param('~svea_name', 'svea2')
        self.DYNAMIC_OBSTACLE_TOPIC = load_param('~dynamic_obstacle_topic', '/dynamic_obstacle')
        self.DEBUG = load_param('~debug', False)
        self.IS_PEDSIM = load_param('~is_pedsim', False)
        self.MEASURE = load_param('~measure', True)
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
        self.pi = PlannerInterface(theta_threshold=0.3)
        # Initialize planner world
        delta, limits, obstacles = self.pi.get_planner_world()
        self.world = AStarWorld(delta=delta, limit=limits, obstacles=np.multiply([delta[0], delta[1], 1], np.array(obstacles)).tolist(), obs_margin=0.2)
        # Initialize planner
        self.planner = AStarPlanner(self.world, [self.state.x, self.state.y], [self.GOAL[0], self.GOAL[1]])

        # Initialize dynamic obstacles simulator
        self.DYNAMIC_OBS = load_param('~dynamic_obstacles', [])
        self.dynamic_obs_simulator = DynamicObstacleSimulator(self.DYNAMIC_OBS, self.DELTA_TIME)
        self.dynamic_obs_simulator.thread_publish()

        # Initialize static unmapped obstacles simulator
        self.STATIC_UNMAPPED_OBS = load_param('~static_unmapped_obstacles', [])
        self.static_unmapped_obs_simulator = StaticUnmappedObstacleSimulator(self.STATIC_UNMAPPED_OBS)
        self.static_unmapped_obs_simulator.publish_obstacle_msg()

        # Initialize social force model helper
        self.sfm_helper = SFMHelper(is_pedsim=self.IS_PEDSIM)

        # Initialize measurement class
        self.measurements = SocialMeasurement(write=True)

        if self.IS_SIM:
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
        self.apf = ArtificialPotentialFieldHelper(svea_name=self.SVEA_NAME)
        self.apf.wait_for_local_costmap()
        # Create vehicle model object
        self.model = BicycleModel(initial_state=self.x0, dt=self.DELTA_TIME)
        # Define variable bounds
        # TODO: maybe avoid going backwards (or penalized it very much)
        x_b = np.array([np.inf, np.inf, 0.7, np.inf])
        u_b = np.array([0.5, np.deg2rad(40)])
        # Create MPC controller object 
        self.controller = SMPC(
            self.model,
            N=self.WINDOW_LEN,
            R=[1, .5],
            x_lb=-x_b,
            x_ub=x_b,
            u_lb=-u_b,
            u_ub=u_b,
            n_states = len(self.x0),
            n_static_obstacles=self.MAX_N_STATIC_OBSTACLES,
            n_dynamic_obstacles=self.MAX_N_DYNAMIC_OBSTACLES,
            n_pedestrians=self.MAX_N_PEDESTRIANS,
            verbose=False
        )

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
        # Compute safe global path
        planner_path = self.planner.create_path()
        # For mich's integration
        #self.pi.get_path_from_topic()
        # Get path 
        # Init path smoother
        self.bspline_smoother = BSpline(planner_path)
        b_spline_path = np.array(self.bspline_smoother.interpolate_b_spline_path(degree=3)).T
        # Get smoothed path and extract social waypoints (other possible nice combination of parameters for path
        # smoothing is interpolate=False and degree=4)
        #b_spline_path = np.array(self.pi.get_social_waypoints(interpolate=True))
        # Create array for MPC reference
        self.path = np.zeros((np.shape(b_spline_path)[0], 4))
        self.path[:, 0] = b_spline_path[:, 0]
        self.path[:, 1] = b_spline_path[:, 1]
        self.path[:, 2] = [self.STRAIGHT_SPEED if abs(curv) < 1e-2 else self.TURN_SPEED for curv in b_spline_path[:, 3]]
        self.path[:, 3] = b_spline_path[:, 2]
        # Init visualize path interface
        self.pi.initialize_path_interface()
        # Re-initialize path interface to visualize on RVIZ socially aware path
        self.pi.set_points_path(self.path[:, 0:2])
        print(f'Social navigation path: {self.path[:, 0:2]} size, {np.shape(self.path)[0]}')

        # If debug mode is on, publish map's representation on RVIZ
        if self.DEBUG:
            self.pi.publish_internal_representation()
        # Publish global path on rviz
        self.pi.publish_path()
        # If measurements are on, save global path
        if self.MEASURE:
            self.measurements.add_global_path(self.path)
            
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

    def get_local_agents(self):
        """
        Function to retrieve agents (i.e. static unmapped obstacles, dynamic obstacles, pedestrians) that are inside the
        local costmap bounds

        :return: local static unmapped obstacles, dynamic obstacles, pedestrians
        :rtype: list[tuple[float]]
        """
        # Get static unmapped obstacles position
        static_unmapped_obs_pos = self.static_unmapped_obs_simulator.obs
        # Initialize array of static unmapped obstacles
        local_static_mpc = np.full((2, self.MAX_N_STATIC_OBSTACLES), -100000.0)
        # Get position of obstacles detected in the local costmap
        static_unmapped_local_obs = self.apf.get_local_obstacles(static_unmapped_obs_pos)
        # Insert them into MPC ready structure
        local_static_mpc[:, 0:np.shape(static_unmapped_local_obs)[0]] = static_unmapped_local_obs.T

        # Initialize array of dynamic obstacles
        local_dynamic_mpc = np.full((4, self.MAX_N_DYNAMIC_OBSTACLES), np.array([[-100000.0, -100000.0, 0, 0]]).T)
        # Acquire mutex
        self.dynamic_obs_simulator.mutex.acquire()
        if (len(self.dynamic_obs_simulator.obs)):
            # Get dynamic obstacle position, v, theta
            dynamic_obs_pose = self.dynamic_obs_simulator.obs[:, 0:4]
            # Release mutex
            self.dynamic_obs_simulator.mutex.release()
            # Get position of obstacles detected in the local costmap
            dynamic_local_obs = self.apf.get_local_obstacles(dynamic_obs_pose)
            # Insert them into MPC structure
            local_dynamic_mpc[:, 0:np.shape(dynamic_local_obs)[0]] = dynamic_local_obs.T
        else:
            # Release mutex as soon as possible
            self.dynamic_obs_simulator.mutex.release()
        
        # Initialize empty pedestrian array
        pedestrians = []
        local_pedestrians_mpc = np.full((4, self.MAX_N_PEDESTRIANS), np.array([[-100000.0, -100000.0, 0, 0]]).T)
        # Acquire mutex
        self.sfm_helper.mutex.acquire()
        # If pedsim is being used
        if self.IS_PEDSIM:
            if self.MEASURE: id = 0
            # For every pedestrian, insert it into the array (necessary since in sfm pedestrians are stored in a dict)
            for p in self.sfm_helper.pedestrian_states:
                pedestrians.append(self.sfm_helper.pedestrian_states[p])
                # If measuring mode is active, add pedestrian pose to file
                if self.MEASURE:
                    self.measurements.add_pedestrian_pose(id, np.array(pedestrians)[-1, :], rospy.get_time())
                    id += 1
        else:
            # If mocap is being used
            pedestrians.append([self.sfm_helper.pedestrian_localizer.state.x, self.sfm_helper.pedestrian_localizer.state.y, self.sfm_helper.pedestrian_localizer.state.v, self.sfm_helper.pedestrian_localizer.state.yaw])
            # If measuring mode is active, add pedestrian pose to file
            if self.MEASURE:
                self.measurements.add_pedestrian_pose(0, np.array(pedestrians)[-1, :], rospy.get_time())
        # Release mutex
        self.sfm_helper.mutex.release()
        # Keep only pedestrians that are in the local costmap
        local_pedestrians = self.apf.get_local_obstacles(pedestrians)
        # Insert them into MPC structure
        local_pedestrians_mpc[:, 0:np.shape(local_pedestrians)[0]] = local_pedestrians.T

        return local_static_mpc, local_dynamic_mpc, local_pedestrians_mpc

    def keep_alive(self):
        """
        Keep alive function based on the distance to the goal and current state of node

        :return: True if the node is still running, False otherwise
        :rtype: boolean
        """
        distance = np.linalg.norm(np.array(self.path[-1, 0:2]) - np.array([self.x0[0], self.x0[1]]))
        return not (rospy.is_shutdown() or distance < self.GOAL_THRESH)

    def run(self):
        """
        Run method
        """
        # Plan a feasible path
        self.plan()
        # Spin until alive
        while self.keep_alive():
            self.spin()
            rospy.sleep(0.1)
        # If measuring mode is active, then close all files, when experiment is done
        if self.MEASURE:
            self.measurements.close_files()
        print('--- GOAL REACHED ---')

    def spin(self):
        """
        Main method
        """
        # Get svea state
        if not self.IS_SIM:
            safe = self.localizer.is_ready
            # Wait for state from localization interface
            self.state = self.localizer.state
            self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        else:
            self.x0 = [self.sim_model.state.x, self.sim_model.state.y, self.sim_model.state.v, self.sim_model.state.yaw]
        print(f'State: {self.x0}')
        # If measuring mode is active, add robot pose to file
        if self.MEASURE:
            self.measurements.add_robot_pose(self.x0, rospy.get_time())

        # Get local static unmapped obstacles, local dynamic obstacles, local pedestrians
        local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc = self.get_local_agents()
        # Compute weight scaling factor, given minimum distance from obstacle (of whatever type, TODO: different
        # sigmoids for different types of obstacles)
        Q = copy(self.Q)
        Q[0:2] *= increasing_sigmoid(self.X_MIN, self.X_MAX, self.Y_MIN_TRAJ, self.Y_MAX_TRAJ, np.min([np.linalg.norm(np.array([self.x0[0:2]]).T - local_static_mpc[0:2, :], axis=0), np.linalg.norm(np.array([self.x0[0:2]]).T - local_dynamic_mpc[0:2, :], axis=0), np.linalg.norm(np.array([self.x0[0:2]]).T - local_pedestrian_mpc[0:2, :], axis=0)]))
        S = self.S * [decreasing_sigmoid(self.X_MIN, self.X_MAX, self.Y_MIN_APF, self.Y_MAX_APF, np.min(np.linalg.norm(np.array([self.x0[0:2]]).T - local_static_mpc[0:2, :], axis=0))), 
                      decreasing_sigmoid(self.X_MIN, self.X_MAX, self.Y_MIN_APF, self.Y_MAX_APF, np.min(np.linalg.norm(np.array([self.x0[0:2]]).T - local_dynamic_mpc[0:2, :], axis=0))),
                      decreasing_sigmoid(self.X_MIN, self.X_MAX, self.Y_MIN_SFM, self.Y_MAX_SFM, np.min(np.linalg.norm(np.array([self.x0[0:2]]).T - local_pedestrian_mpc[0:2, :], axis=0)))]
        print(self.Q, Q)
        print(self.S, S)
        # Get next waypoint index (by computing offset between robot and each point of the path), wrapping it in case of
        # index out of bounds
        self.waypoint_idx = np.minimum(np.argmin(np.linalg.norm(self.path[:, 0:2] - np.array([self.x0[0], self.x0[1]]), axis=1)) + 1, np.shape(self.path)[0] - 1)

        # If there are not enough waypoints for concluding the path, then fill in the waypoints array with the desiderd
        # final goal
        if self.waypoint_idx + self.WINDOW_LEN + 1 >= np.shape(self.path)[0]:
            last_iteration_points = self.path[self.waypoint_idx:, :]
            while np.shape(last_iteration_points)[0] < self.WINDOW_LEN + 1:
                last_iteration_points = np.vstack((last_iteration_points, self.path[-1, :]))
            u, predicted_state = self.controller.get_ctrl(self.x0, last_iteration_points[:, :].T, local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc, Q, S)
        else:
            u, predicted_state = self.controller.get_ctrl(self.x0, self.path[self.waypoint_idx:self.waypoint_idx + self.WINDOW_LEN + 1, :].T, local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc, Q, S)

        # Get optimal velocity (by integrating once the acceleration command and summing it to the current speed) and
        # steering controls
        if self.IS_SIM:
            velocity = u[0, 0] * self.DELTA_TIME + self.x0[2]
        else:
            velocity = u[0, 0] * self.DELTA_TIME_REAL + self.x0[2]
        steering = u[1, 0]
        print(f'Optimal control (acceleration, velocity, steering): {u[0, 0], velocity, steering}')
        
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
import numpy as np
import casadi
from svea.models.generic_mpc import GenericModel

class MPC(object):
    # Repulsive force coefficient
    K_R = 1

    def __init__(self, model: GenericModel, x_lb, x_ub, u_lb, u_ub, n_obstacles, Q, R, S, N=7, apply_input_noise=False, apply_state_noise=False, verbose=False):
        """
        Init method for MPC class

        :param model: kinodynamic model of the systems
        :type model: GenericModel
        :param x_lb: state variables lower bounds
        :type x_lb: Tuple[float]
        :param x_ub: state variables upper bounds
        :type x_ub: Tuple[float]
        :param u_lb: input variables lower bounds
        :type u_lb: Tuple[float]
        :param u_ub: input variables upper bounds
        :type u_ub: Tuple[float]
        :param N: number of inputs to be predicted, defaults to 7
        :type N: int, optional
        :param apply_input_noise: True the model applies Gaussian noise the the input variables, defaults to False
        :type apply_input_noise: bool, optional
        :param apply_state_noise: True the model applies Gaussian noise the the state variables, defaults to False
        :type apply_state_noise: bool, optional
        """
        self.model = model
        # Get model's number of states and inputs
        self.n_states = len(self.model.dae.x)
        self.n_inputs = len(self.model.dae.u)

        # Get max number of obstacles
        self.n_obs = n_obstacles

        # Check that there are enough lower and upper bounds for each state/input variable
        assert self.n_states == len(x_lb), f'Number of lower bounds does not correspond to states number, number of states: {self.n_states}, number of lower bounds: {len(x_lb)}'
        assert self.n_states == len(x_ub), f'Number of lower bounds does not correspond to states number, number of states: {self.n_states}, number of lower bounds: {len(x_ub)}'
        assert self.n_inputs == len(u_lb), f'Number of lower bounds does not correspond to states number, number of states: {self.n_inputs}, number of lower bounds: {len(u_lb)}'
        assert self.n_inputs == len(u_ub), f'Number of lower bounds does not correspond to states number, number of states: {self.n_inputs}, number of lower bounds: {len(u_ub)}'
        # Get matrices of weights for cost function
        assert self.n_states == np.shape(Q)[0], f'Number of weights in states weights matrix Q does not correspond number of states, number of states: {self.n_states}, number of weights: {np.shape(Q)[0]}'
        assert self.n_inputs == np.shape(R)[0], f'Number of weights in inputs weights matrix R does not correspond number of inputs, number of inputs: {self.n_inputs}, number of weights: {np.shape(R)[0]}'
        
        # Get weights matrices
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.S = casadi.diag(S)
        # Get number of controls to be predictes 
        self.N = N
        # Get initial state
        self.initial_state = self.model.initial_state
        

        # Get states lower and upped bounds
        self.x_lb = casadi.DM(x_lb)
        self.x_ub = casadi.DM(x_ub)
        # Get inputs lower and upper bounds
        self.u_lb = casadi.DM(u_lb)
        self.u_ub = casadi.DM(u_ub)

        # Get noise settings
        self.apply_input_noise = apply_input_noise
        self.apply_state_noise = apply_state_noise

        # Define optimizer variables 
        self.opti = None
        self.x = None
        self.u = None
        self.reference_state = casadi.DM.zeros(self.n_states, 1)
        self.obs_position = None

        # Cost function related varirables
        self.state_diff = []
        self.angle_diff = []
        self.F_r = []
        self.cost = 0

        # Verbose option
        self.verbose = verbose

        # Build optimizer 
        self._build_optimizer()

    def _build_optimizer(self):
        """
        Method to build the optimizer problem
        """
        # Define optimizer environment
        self.opti = casadi.Opti()

        # Set optimizer variables
        self._optimizer_variables()
        # Set optimizer parameters
        self._optimizer_parameters()
        # Set optimizer cost function
        self._optimizer_cost_function()
        # Set optimizer constraints
        self._optimizer_constraints()

        # TODO: optimizer options
        p_opts = {
            "verbose": self.verbose,
            "expand": True,
            "print_in": False,
            "print_out": False,
            "print_time": False}
        s_opts = {"max_iter": 150,
                  "print_level": 1,
                  "fixed_variable_treatment": "make_constraint",
                  "barrier_tol_factor": 1,
                  }

        self.opti.solver('ipopt', p_opts, s_opts)

    def _optimizer_variables(self):
        """
        Method used to set optimizer variables
        """
        # Define optimizer variables
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(self.n_inputs, self.N)
        # Slack variable
        #self.slack = self.opti.variable(self.n_states, 1)

    def _optimizer_parameters(self):
        """
        Method used to set optimizer parameters
        """
        # Define optimizer parameters (constants)
        # Rows as the number of state variables to be kept into accout, columns how many timesteps
        self.reference_state = self.opti.parameter(self.n_states, 1)
        self.initial_state = self.opti.parameter(self.n_states, 1)
        self.obs_position = self.opti.parameter(2, self.n_obs)

    def _optimizer_cost_function(self):
        """
        Method used to set optimizer cost function
        """
        self.cost = 0
        # Extract dimension of reference state (every state variable minus the heading)
        reference_dimension = np.shape(self.reference_state)[0] - 1
        for k in range(self.N + 1):
            # Predicted state to be as close as possible to reference one
            self.state_diff.append(self.x[:reference_dimension, k] - self.reference_state[:reference_dimension, 0])
            self.cost += self.state_diff[k].T @ self.Q[:reference_dimension, :reference_dimension] @ self.state_diff[k]
            # Compute angle diff as the heading difference between current heading angle of robot and angle between
            # robot's position and waypoint position
            #self.angle_diff.append(self.x[-1, k] - casadi.arctan2((self.reference_state[1, k] - self.x[1, k]), (self.reference_state[0, k] - self.x[0, k])))
            #self.cost += self.angle_diff[k]**2 * self.Q[-1, -1]
            # In this way heading from current waypoint to next one, has to be computed for each waypoint (don't like it
            # much, but it is much more robust)
            self.angle_diff.append(np.pi - casadi.norm_2(casadi.norm_2(self.x[-1, k] - self.reference_state[-1, 0]) - np.pi))
            #self.angle_diff.append(self.reference_state[-1, 0] - self.x[-1, k])
            self.cost += self.angle_diff[k]**2 * self.Q[-1, -1]

            rep_force = 0
            for i in range(self.n_obs):
                rep_force += 2 * casadi.exp(-(((self.x[0, k] - self.obs_position[0, i]) ** 2 / 2) + ((self.x[1, k] - self.obs_position[1, i]) ** 2 / 2)))
            self.F_r.append(rep_force)
            self.cost += self.S * self.F_r[k]

            """rep_force = 0
            for i in range(self.n_obs):
                rep_force += 2 * casadi.exp(-(((self.x[0, k] - self.obs_position[0, i]) ** 2 / 2) + ((self.x[1, k] - self.obs_position[1, i]) ** 2 / 2)))
            self.F_r.append(rep_force)
            self.cost += self.S * self.F_r[k]

            j_obs = 0
            for i in range(self.n_obs):
                j_obs += (2 * self.x[2, k]) / ((self.obs_position[0, i] - self.x[0, k])**2 + (self.obs_position[1, i] - self.x[1, k])**2 + 0.0001)
            self.F_r.append(j_obs)
            self.cost += self.S * self.F_r[k]

            exp = 0
            for i in range(self.n_obs):
                exp += casadi.if_else(self.obs_position[0, i] != -100000.0, casadi.sqrt((self.x[0, k] - self.obs_position[0, i]) ** 2 + (self.x[1, k] - self.obs_position[1, i]) ** 2), 0, True)
            exp = casadi.if_else(exp == 0, 100, exp)
            self.F_r.append(0.5 * self.K_R * casadi.exp(-exp))
            self.cost += self.S * self.F_r[k]"""

            if k < self.N:
                # Weight and add to cost the control effort
                self.cost += self.u[:, k].T @ self.R @ self.u[:, k]
        # Add slack variable to the cost
        #self.cost += 1e2 * self.slack.T @ self.slack

        # Set cost function for the optimizer
        self.opti.minimize(self.cost)

    def _optimizer_constraints(self):
        """
        Method used to define optimizer constraints
        """
        # Set kinodynamic constraints given by the model for every control input to be predicted
        for t in range(self.N):
            # Generate next state given the control 
            x_next, _ = self.model.f(self.x[:, t], self.u[:, t], apply_input_noise=self.apply_input_noise, apply_state_noise=self.apply_state_noise)
            self.opti.subject_to(self.x[:, t + 1] == x_next)
            # With slack variable
            #self.opti.subject_to(self.x[:, t + 1] == x_next + self.slack)

        # Set state bounds as optimizer constraints
        self.opti.subject_to(self.opti.bounded(self.x_lb, self.x, self.x_ub))
        # Set input bounds as optimizer constraints
        self.opti.subject_to(self.opti.bounded(self.u_lb, self.u, self.u_ub))
        # Set initial state as optimizer constraint
        self.opti.subject_to(self.x[:, [0]] == self.initial_state)

    def get_ctrl(self, initial_state, reference_state, obs_pos):
        """
        Function to solve optimizer problem and get control from MPC, given initial state and reference state

        :param initial_state: initial state
        :type initial_state: Tuple[float]
        :param reference_state: reference state
        :type reference_state: Tuple[float]
        :return: _description_
        :rtype: _type_
        """
        # Set optimizer values for both initial state and reference states
        self.opti.set_value(self.initial_state, initial_state)
        self.opti.set_value(self.reference_state, reference_state)
        self.opti.set_value(self.obs_position, obs_pos)
        # Solve optimizer problem if it is feasible
        try: 
            self.opti.solve()
        except RuntimeError as e:
            print(f'Cost: {self.opti.debug.value(self.cost)}')
            for angle, state, r_force in zip(self.angle_diff, self.state_diff, self.F_r):
                print(f'Angle diff: {self.opti.debug.value(angle)}')
                print(f'State diff: {self.opti.debug.value(state)}')
                print(f'Repulsive force: {self.opti.debug.value(r_force)}')
            self.opti.debug.show_infeasibilities()
            #self.opti.debug.x_describe()
            #self.opti.debug.g_describe()
            raise(e)
        if self.verbose:
            #print(f'Predicted control sequence: {self.opti.value(self.u[:, :])}')
            print(f'Cost: {self.opti.debug.value(self.cost)}')
            for angle, state, r_force in zip(self.angle_diff, self.state_diff, self.F_r):
                print(f'Angle diff: {self.opti.debug.value(angle)}')
                print(f'State diff: {self.opti.debug.value(state)}')
                print(f'Repulsive force: {self.opti.debug.value(r_force)}')
        #for r_force in self.F_r:
        #    print(f'MPC Repulsive force: {self.opti.debug.value(r_force)}')
        print(f'MPC Cost: {self.opti.debug.value(self.cost)}')
        #print(f'MPC State: {self.opti.debug.value(self.x)}')
        #print(f'MPC Reference: {self.opti.debug.value(self.reference_state)}')
        #print(f'MPC Obs: {self.opti.debug.value(self.obs_position)}')
        # Get first control generated (not predicted ones)
        u_optimal = np.expand_dims(self.opti.value(self.u[:, 0]), axis=1)
        # Get new predicted position
        x_pred = self.opti.value(self.x)
        return u_optimal, x_pred

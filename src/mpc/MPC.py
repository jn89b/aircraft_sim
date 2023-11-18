import casadi as ca
import numpy as np
from src.config import Config

class ModelPredictiveControl():
    def __init__(self, mpc_params):

        params_list = ['model', 'dt_val', 'N', 'Q', 'R']
        for param in params_list:
            if param not in mpc_params:
                raise Exception(f'{param} not in mpc_params')

        self.mpc_params = mpc_params
        self.model = self.mpc_params['model']
        #check if model is a casadi function
        self.f = self.model.f
        # if not isinstance(self.model.function, ca.Function):
        #     self.f = self.model.f
        # else:
        #     self.f = self.model.function
        self.dt_val = self.mpc_params['dt_val']
        self.N = self.mpc_params['N']
        self.Q = self.mpc_params['Q']
        self.R = self.mpc_params['R']

        self.n_states = self.model.n_states
        self.n_controls = self.model.n_controls

        self.S =  2.0#obstacle avoidance weight
        self.cost_fn = 0
        self.solver = None

    def returnTimeVector(self):
        time_vec = np.arange(0, self.N * self.dt_val, self.dt_val)


    def initDecisionVariables(self):
        #decision variables
        """intialize decision variables for state space models"""
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)

        #column vector for storing initial and target locations
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )

        self.g = []
        self.g = self.X[:,0] - self.P[:self.n_states]
        
    def defineBoundaryConstraints(self):
        """define bound constraints of system"""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']
        
        #function to turn decision variables into one long row vector
        self.pack_variables_fn = ca.Function('pack_variables_fn', self.variables_list, 
                                             [self.OPT_variables], self.variables_name, 
                                             ['flat'])
        
        #function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function('unpack_variables_fn', [self.OPT_variables], 
                                               self.variables_list, ['flat'], self.variables_name)

        ##helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)

    def addAdditionalConstraints(self):
        """add additional constraints to system"""
        

    def computeCost(self):
        #tired of writing self
        #dynamic constraints 
        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        
        #compute cost function for error between current and target state
        # self.cost_fn = (self.X[:, -1] - P[n_states:]).T @ Q @ (self.X[:, -1] - P[n_states:])  \
        #                 + self.U[:, -1].T @ R @ self.U[:, -1]       

        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
            
            #penalize states and controls for now, can add other stuff too
            self.cost_fn = self.cost_fn \
                + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
                + controls.T @ R @ controls                 

            # self.cost_fn =             
            ##Runge Kutta
            k1 = self.f(states, controls)
            k2 = self.f(states + self.dt_val/2*k1, controls)
            k3 = self.f(states + self.dt_val/2*k2, controls)
            k4 = self.f(states + self.dt_val * k3, controls)
            state_next_RK4 = states + (self.dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, state_next - state_next_RK4) #dynamic constraints
      
        if Config.OBSTACLE_AVOID:
            for k in range(self.N):
                #penalize obtacle distance
                x_pos = self.X[0,k]
                y_pos = self.X[1,k]                
                obs_distance = ca.sqrt((x_pos - Config.OBSTACLE_X)**2 + \
                                        (y_pos - Config.OBSTACLE_Y)**2) 

    
                obs_constraint = -obs_distance + ((Config.ROBOT_DIAMETER/2) + \
                    (Config.OBSTACLE_DIAMETER/2))

                self.cost_fn = self.cost_fn + (self.S* obs_distance)

                self.g = ca.vertcat(self.g, obs_constraint) 

        if Config.MULTIPLE_OBSTACLE_AVOID:
            for obstacle in Config.OBSTACLES:
                obs_x = obstacle[0]
                obs_y = obstacle[1]
                obs_diameter = obstacle[2]

                for k in range(self.N):
                    #penalize obtacle distance
                    x_pos = self.X[0,k]
                    y_pos = self.X[1,k]                
                    obs_distance = ca.sqrt((x_pos - obs_x)**2 + \
                                            (y_pos - obs_y)**2)
                    

                    obs_constraint = -obs_distance + ((Config.ROBOT_DIAMETER) + \
                        (obs_diameter/2)) 

                    self.g = ca.vertcat(self.g, obs_constraint)
                
                    if obstacle == [Config.GOAL_X, Config.GOAL_Y]:
                        continue
                    
                    self.cost_fn = self.cost_fn + (self.S* obs_distance)


    def initSolver(self):
        
        nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }
        
        # solve the optimization problem
        solver_opts = {
            'ipopt': {
                'max_iter': Config.MAX_ITER,
                'print_level': Config.PRINT_LEVEL,
                'acceptable_tol': Config.ACCEPT_TOL,
                'acceptable_obj_change_tol': Config.ACCEPT_OBJ_TOL,
                # 'hessian_approximation': 'limited-memory',  # Set Hessian approximation method here
            },
            'print_time': Config.PRINT_TIME
        }

        #create solver
        self.solver = ca.nlpsol('solver', 'ipopt', 
            nlp_prob, solver_opts)
    
    def reinitStartGoal(self, start, goal):
        self.state_init = ca.DM(start)   # initial state
        self.state_target = ca.DM(goal)  # target state

    # def moveObstacle(self, x, y):
    #     """move obstacle to new position"""
    #     self.obstacle_x = x + self.dt_val * Config.OBSTACLE_VX
    #     self.obstacle_y = y + self.dt_val * Config.OBSTACLE_VY 
    
    def shiftTimestep(self,step_horizon, t_init, state_init, u, f):
        """
        we shift the time horizon over one unit of the step horizon
        reinitialize the time, new state (which is next state), and 
        the control parameters
        """
        f_value = f(state_init, u[:, 0]) #calls out the runge kutta
        next_state = ca.DM.full(state_init + (step_horizon * f_value))

        #shift time and controls
        next_t = t_init + step_horizon
        next_control = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        return next_t, next_state, next_control


    def solveMPCRealTimeStatic(self, start, goal, controls, init_solver:bool=True):
        """solve the mpc based on initial and desired location"""
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        
        self.state_init = ca.DM(start)        # initial state
        self.state_target = ca.DM(goal)  # target state
        print("state_init: ", self.state_init)
        self.controls = ca.DM(controls)  # initial control
        # self.t0 = t0
        #self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control 
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full
        self.u0 = ca.repmat(self.controls, 1, self.N)             # initial control full

        """Jon's advice consider velocity of obstacle at each knot point"""
        #moving target in the y direction
        self.state_target = ca.DM(goal) 

        if init_solver == True:
            self.initSolver()

        if Config.RADAR_AVOID == True:
            # constraints lower bound added
            n_radar_networks = 1
            num_constraints = n_radar_networks * self.N
            lbg =  ca.DM.zeros((self.n_states*(self.N+1)+num_constraints, 1))
            # -infinity to minimum marign value for obs avoidance  
            lbg[self.n_states*self.N+n_states:] = -ca.inf # stay under 0.5 probability of detection 
            
            # constraints upper bound
            ubg  =  ca.DM.zeros((self.n_states*(self.N+1)+num_constraints, 1))
            #rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
            ubg[self.n_states*self.N+n_states:] = 0.5

        elif Config.OBSTACLE_AVOID:
            print("Obstacle Avoidance Enabled")
            """NEEED TO ADD OBSTACLES IN THE LBG AND UBG"""
            # constraints lower bound added 
            lbg =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))
            # -infinity to minimum marign value for obs avoidance  
            lbg[self.n_states*self.N+n_states:] = -ca.inf 
            
            # constraints upper bound
            ubg  = ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))
            #rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
            ubg[self.n_states*self.N+n_states:] = -Config.BUFFER_DISTANCE

        elif Config.MULTIPLE_OBSTACLE_AVOID:
            print("Multiple Obstacle Avoidance Enabled")
            """NEEED TO ADD OBSTACLES IN THE LBG AND UBG"""
            # constraints lower bound added 
            num_constraints = Config.N_OBSTACLES * self.N
            lbg =  ca.DM.zeros((self.n_states*(self.N+1)+num_constraints, 1))
            # -infinity to minimum marign value for obs avoidance  
            lbg[self.n_states*self.N+n_states:] = -ca.inf 
            
            # constraints upper bound
            ubg  =  ca.DM.zeros((self.n_states*(self.N+1)+num_constraints, 1))
            #rob_diam/2 + obs_diam/2 #adding inequality constraints at the end 
            ubg[self.n_states*self.N+n_states:] = -Config.BUFFER_DISTANCE

        else:
            lbg = ca.DM.zeros((self.n_states*(self.N+1), 1))
            ubg  =  ca.DM.zeros((self.n_states*(self.N+1), 1))

        args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }

        #this is where you can update the target location
        args['p'] = ca.vertcat(
            self.state_init,    # current state
            self.state_target   # target state
        )
        
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(self.X0, n_states*(self.N+1), 1),
            ca.reshape(self.u0, n_controls*self.N, 1)
        )

        #this is where we solve
        self.sol = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        #unpack as a matrix
        self.u = ca.reshape(self.sol['x'][self.n_states * (self.N + 1):], 
                            self.n_controls, self.N)
        
        self.X0 = ca.reshape(self.sol['x'][: n_states * (self.N+1)], 
                                self.n_states, self.N+1)        
        
        #return the controls and states of the system
        return (self.u, self.X0)
        

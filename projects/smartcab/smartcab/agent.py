import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, agent_type=None, n_training=20, lower_bound=0.01, optimized_env=False):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.optimized_env = optimized_env
        self.training_trials = 0
        self.testing_trials = 0

        self.n_training = n_training
        self.lower_bound = lower_bound

        lin = lambda n, lb, x: float(n - x) / (n - 1)
        inv = lambda n, lb, x: 1. / ((1 / lb - 1) / (n - 1) * (x - 1) + 1)
        exp = lambda n, lb, x: math.exp(math.log(lb) / (n - 1) * (x - 1))
        cos = lambda n, lb, x: math.cos(math.pi / (2 * (n - 1)) * (x - 1))
        cos_sqrt = lambda n, lb, x: math.sqrt(math.cos(math.pi / (2 * (n - 1)) * (x - 1)))
        mirror = lambda f, n, lb, x: 1 - f(n, lb, n - x + 1)

        if agent_type == "neutral":
            # Neutral - Linear 1 -> 0
            self.epsilon_func = lambda x: lin(n_training, lower_bound, x)
        elif agent_type == "explore-5":
            # Explore 5 - Constant 1
            self.epsilon_func = lambda x: 1
        elif agent_type == "explore-4":
            # Explore 4 - Inverse (1 - lower_bound -> 0)
            self.epsilon_func = lambda x: mirror(inv, n_training, lower_bound, x)
        elif agent_type == "explore-3":
            # Explore 3 - Exponential (1 - lower_bound -> 0)
            self.epsilon_func = lambda x: mirror(exp, n_training, lower_bound, x)
        elif agent_type == "explore-2":
            # Explore 2 - Trigonometric sqrt explorative (1 -> 0)
            self.epsilon_func = lambda x: cos_sqrt(n_training, lower_bound, x)
        elif agent_type == "explore-1":
            # Explore 1 - Trigonometric explorative (1 -> 0)
            self.epsilon_func = lambda x: cos(n_training, lower_bound, x)
        elif agent_type == "exploit-1":
            # Exploit 1 - Trigonometric exploitative (1 -> 0)
            self.epsilon_func = lambda x: mirror(cos, n_training, lower_bound, x)
        elif agent_type == "exploit-2":
            # Exploit 2 - Trigonometric sqrt exploitative (1 -> 0)
            self.epsilon_func = lambda x: mirror(cos_sqrt, n_training, lower_bound, x)
        elif agent_type == "exploit-3":
            # Exploit 3 - Exponential (1 -> lower_bound)
            self.epsilon_func = lambda x: exp(n_training, lower_bound, x)
        elif agent_type == "exploit-4":
            # Exploit 4 - Inverse  (1 -> lower_bound)
            self.epsilon_func = lambda x: inv(n_training, lower_bound, x)
        elif agent_type == "exploit-5":
            # Exploit 5 - Constant 0
            self.epsilon_func = lambda x: 0
        else:
            raise AssertionError()


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        # Else update epsilon using a decay function of your choice
        if testing:
            self.testing_trials += 1
            self.epsilon = 0
            self.alpha = 0
        else:
            self.training_trials += 1
            self.epsilon = self.epsilon_func(self.training_trials)

            # self.epsilon = 1 - 1. / ((1 / low - 1) / (n_training - 1) * (self.n_training - self.training_trials) + 1)
            # self.epsilon = 1 - math.exp(- math.log(low) / (self.n_training - 1) * (self.n_training - self.training_trials))
            # self.epsilon = 1 - math.sin(math.pi / (2 * (self.n_training - 1)) * (self.training_trials - 1))
            # self.epsilon = 1 - math.sqrt(math.sin(math.pi / (2 * (self.n_training - 1)) * (self.training_trials - 1)))

        return None


    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ##################
        ## TO DO (DONE) ##
        ##################

        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # DONE: Set 'state' as a tuple of relevant data for the agent
        # 1st attempt: I believed you had to wait for oncoming traffic...
        #state = (waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        # 2nd attempt: Apparently one should ignore oncoming traffic when the lights are green and just act...
        #state = (waypoint, inputs['light'])
        # 3rd attempt: But apparently only ignore oncoming traffic then, not when travelling right with a red light.
        #state = (waypoint, inputs['light'], inputs['left'])
        # 4th attempt: But apparently this was fixxed, as my initial interpretation was correct and the environment was faulty, and fixxed
        #              in this commit: https://github.com/udacity/machine-learning/commit/912f9edbb2c6647039cf71c42f0f075d842b273e
        state = (waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ##################
        ## TO DO (DONE) ##
        ##################
        # DONE: Calculate the maximum Q-value of all actions for a given state
        maxQ = max(self.Q[state].values())

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ##################
        ## TO DO (DONE) ##
        ##################
        # DONE: 
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        # Then, for each action available, set the initial Q-value to 0.0
        initial_Q = 0

        if self.optimized_env:
            initial_Q = 10

        if self.learning and not self.Q.has_key(state):
            self.Q[state] = {action:initial_Q for action in self.env.valid_actions}

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        
        ##################
        ## TO DO (DONE) ##
        ##################
        # DONE: When not learning, choose a random action
        # DONE: When learning, choose a random action with 'epsilon' probability
        # DONE: Otherwise, choose an action with the highest Q-value for the current state
        # DONE: Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".

        # Default case: random action is chosen unless learning and not an epsilon likely event occurs
        action = self.valid_actions[random.randint(0, 3)]

        if self.learning and not random.random() < self.epsilon:
            maxQ = self.get_maxQ(state)
            maxQ_actions = [a for (a, Q) in self.Q[state].iteritems() if Q == maxQ]
            action = random.choice(maxQ_actions)

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning.

            NOTE: We were in a state, we choose an action, we got a reward.
            """

        ##################
        ## TO DO (DONE) ##
        ##################
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            # NOTE: That 0 could be replaced with a discount factor multiplied with the
            #       maxQ of the state we landed in. This function does not have access
            #       to the state we landed in though.
            self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + 0)

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return


    def get_optimal_action(self, state):
        """ Returns the optimal policy action for a given state, based on hardcoded theory. """

        waypoint = state[0]
        light = state[1]
        oncoming = state[2]
        left = state[3]

        if light == 'green':
            action = waypoint
            if waypoint == 'left' and (oncoming == 'forward' or oncoming == 'right'):
                action = None
        else:
            action = None
            if waypoint == 'right' and not left == 'forward':
                action = 'right'
        
        return action
        

def run(agent_type=None, n_training=20, n_test=10, tolerance=0.05, epsilon=1, alpha=0.5, lower_bound=0.05, enforce_deadline=False, optimized_env=False):
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(optimized_env=optimized_env)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, agent_type=agent_type, n_training=n_training, epsilon=epsilon, alpha=alpha, optimized_env=optimized_env)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=enforce_deadline)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    #   log_name     - set to change the log file name
    sim = Simulator(env, update_delay=0.001, display=False, log_metrics=True, optimized_env=optimized_env, log_name='sim_'+agent_type+'_train-'+str(n_training)+'_test-'+str(n_test)+'_alpha-'+str(alpha)+'_enforce-deadline-'+str(enforce_deadline)+'_optimized-env-'+str(optimized_env))
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    #   n_training - the minimum number of training trials
    sim.run(n_training=n_training, n_test=n_test, tolerance=tolerance)

    print "Custom simulation report."
    print "Training trials: {}, testing trials: {}".format(agent.training_trials, agent.testing_trials)


if __name__ == '__main__':

    if False:
        run(agent_type='exploit-1')
    else:
        agents = ['exploit-5'] # agents = ['explore-4', 'explore-3', 'explore-2', 'explore-1', 'neutral', 'exploit-1', 'exploit-2', 'exploit-3', 'exploit-4']
        enforce_deadlines = [True]
        optimized_env = [True]
        n_training = [100, 250, 500, 1000]
        n_test = [100]
        alpha = [1]
        tolerance = 2
        lower_bound = 0.01

        for o in optimized_env:
            for ed in enforce_deadlines:
                for a in alpha:
                    for t in n_test:
                        for n in n_training:
                            for agent in agents:
                                print "Starting simulation of '{}', with n_training: {}, n_test: {}, tolerance: {}".format(agent, n_training, n_test, tolerance)
                                run(agent_type=agent, n_training=n, n_test=t, alpha=a, tolerance=tolerance, lower_bound=lower_bound, enforce_deadline=ed, optimized_env=o)
                                print "Finished simulation of '{}', with n_training: {}, n_test: {}, tolerance: {}".format(agent, n_training, n_test, tolerance)

# high Q inits, deterministic rewards
# -> alpha 1, exploit-5
# 0 Q inits, deterministic rewards
# -> alpha 1, neutral
# non-deterministic rewards
# -> alpha 0.5, neutral



# Kung fu solution:
# initialize high rewards, use deterministic rewards (no random initialization of reward between 1 and -1, and without penalty, exploit nonstop, alpha = 1, 200 runs (that goes quick since high performance movements are made)

# If you cant initialize high rewards, but have to make due with a 0 initialization...
# Of if you cant use deterministic rewards
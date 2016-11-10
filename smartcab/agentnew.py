import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.valid_actions = env.valid_actions
        self.q_dict = {}
        self.learning_rate = 0.1
        self.discount_rate = 0.7
        
        #statistics to analyse performance
        self.total_wins = 0
        self.trial_infractions = 0
        self.infractions_record = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.infractions_record.append(self.trial_infractions)
        self.trial_infractions = 0

    def update(self, t):
 
        # Next way_point from route planner, also displayed by simulator. 
        # Shortest distance to goal. Does not obey the rules of traffic.
        self.next_waypoint = self.planner.next_waypoint()

        #Senses the state of the smartcab
        inputs = self.env.sense(self)

        #Gets the deadline for the current trial
        deadline = self.env.get_deadline(self)

        # Updates state
        state = self.set_state(inputs)

        # TODO: Select action according to your policy
        action = self.select_optimal_action(state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_q_dict(reward, state, action)

        #save some statistics to analyze how well the model is doing.
        self.save_stats(reward)

        #print an update
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def performace_report(self, n_trials):

    	print '\n'+ 25*'*' + "FINAL REPORT:" + 25*'*'
    	print 'AMOUNT OF TIMES REACHED GOAL:', self.total_wins
    	print 'TRAFFIC INFRACTIONS RECORD:', self.infractions_record
    	print 'AMOUNT OF TIMES GOAL NOT REACHED', (n_trials - self.total_wins)
    	print 'TOTAL AMOUNT OF TRAFFIC INFRACTIONS:', sum(self.infractions_record)
    	print 60*'*'+'\n'

    def set_state(self, inputs):
   	  	return {'light': inputs['light'],
        	    'oncoming': inputs['oncoming'],
        		'left': inputs['left'],
        		'next_waypoint': self.next_waypoint
        		}    
    def save_stats(self,reward):
    	"""records how many times the simulator reaches the goal"""
    	if reward >= 5:
    		self.total_wins += 1
    	if reward <= -1:
    		self.trial_infractions += 1
    	else:
    		pass

    def select_optimal_action(self, state):
    	"""
    	Returns the optimal action based on the max q_value in q_dict for the 
    	valid_actions.
    	"""

        #initialize best_action and best_value    
        optimal_action = self.valid_actions[0]
        optimal_value = 0

        # check each of the valid actions for the current state in the Q-dict and return the 
        # action that has the highest q_value.
        # If all of the q_values are the same, return a random choice.
        for action in self.valid_actions:
            cur_value = self.get_q_value(state, action)
            if cur_value > optimal_value:
                optimal_action = action
                optimal_value = cur_value
            elif cur_value == optimal_value:
                optimal_action = random.choice(self.valid_actions)
        return optimal_action
    	

    
    def update_q_dict(self, reward, state, action):
    	"""Updates q_value in q_dict using learning and discount rate"""
    	q_key = self.get_q_key(state, action)
        
        #finds new_state after performing action.
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = self.set_state(inputs)

        #assigns new value to q_dict.
        x = self.get_q_value(state, action) 
        V = reward + (self.discount_rate * self.find_max_q(new_state))
        new_q_value = V + (self.learning_rate * (V - x))
        self.q_dict[q_key] = new_q_value
        

    def find_max_q(self,state):
    	"""returns max q_value in q_dict for given state"""
    	max_value = None
        for action in self.valid_actions:
            cur_value = self.get_q_value(state, action)
            if max_value is None or cur_value > max_value:
                max_value = cur_value
               
        return max_value


    def get_q_value(self, state, action):
    	"""returns the q_value, given a state action pair"""
        q_key = self.get_q_key(state, action)
        if q_key in self.q_dict:
            return self.q_dict[q_key]
        return 0 	

    def get_q_key(self, state, action):
    	"""Returns the q_key given a state, action pair"""
    	return '{}-{}-{}-{}-{}'.format(state['light'], state['oncoming'], state['left'], state['next_waypoint'], action)


def run():
    """Run the agent for a finite number of trials."""
    number_of_trials = 100
    print 'RUNNING SIMULATION FOR {} TRIALS...'.format(number_of_trials)
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=number_of_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    a.performace_report(number_of_trials)

if __name__ == '__main__':
    run()

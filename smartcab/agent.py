import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # super uses the .__init__ function from the super class "Agent"
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.state = {}
        self.learning_rate = 0.6
        self.exploration_rate = 0.1
        self.exploration_degradation_rate = 0.001
        self.discount_rate = 0.4
        self.q_values = {}
        self.valid_actions = env.valid_actions

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = self.set_state(inputs)
        

        # TODO: Select action according to your policy. This will be performed in the next iteration sim.run()
        action = self.choose_action_from_policy(self.state)

        # Execute action and get reward from act.
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.update_q_value(self.state, action, reward)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def set_state(self, inputs):
      return {
        "light": inputs["light"],
        "oncoming": inputs["oncoming"],
        "left": inputs["left"],
        "direction": self.next_waypoint
      }

    def choose_action_from_policy(self, state):
        #simulated annealing?
        if random.random() < self.exploration_rate:
            self.exploration_rate -= self.exploration_degradation_rate
            return random.choice(self.valid_actions)

        #initialize best_action and best_value    
        best_action = self.valid_actions[0]
        best_value = 0

        #check each of the valid actions for the current state in the Q-table and return the highest q-value.
        #If all of the q-values are the same, return a random choice.
        for action in self.valid_actions:
            cur_value = self.q_value_for(state, action)
            if cur_value > best_value:
                best_action = action
                best_value = cur_value
            elif cur_value == best_value:
                best_action = random.choice([best_action, action])
        return best_action

    def max_q_value(self, state):
        max_value = None
        for action in self.valid_actions:
            cur_value = self.q_value_for(state, action)
            if max_value is None or cur_value > max_value:
                max_value = cur_value
        return max_value

    def q_value_for(self, state, action):
        q_key = self.q_key_for(state, action)
        if q_key in self.q_values:
            return self.q_values[q_key]
        return 0

    def update_q_value(self, state, action, reward):
        q_key = self.q_key_for(state, action)
        cur_value = self.q_value_for(state, action)
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = self.set_state(inputs)
        learned_value = reward + (self.discount_rate * self.max_q_value(new_state))
        new_q_value = cur_value + (self.learning_rate * (learned_value - cur_value))
        self.q_values[q_key] = new_q_value

    def q_key_for(self, state, action):
        return "{}-{}-{}-{}-{}".format(state["light"], state["direction"], state["oncoming"], state["left"], action)



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    # print "CONCLUSION REPORT"
    # print "WINS: {}".format(e.wins)
    # print "LOSSES: {}".format(e.losses)
    # print "INFRACTIONS: {}".format(e.infractions)

if __name__ == '__main__':
    run()
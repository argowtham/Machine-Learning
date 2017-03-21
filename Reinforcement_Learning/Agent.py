from Reinforcement_Learning import GridSquare


class Agent(object):
    """An agent whose aim is to reach the goal state. It has the following properties:

    Attributes:
        :current_location - A list indicating the current location of the agent
        :best_action: - A string indicating the best action from the current state
        :profit: - Profit of the agent so far
        :origin: - A GridSquare Object ([0,0]) as the agent starts here
        :move_cost: - Cost incurred by agent for moving from one square to other
        :gamma: - discount rate
        """

    def __init__(self):
        self.current_location = None
        self.actions = None
        self.profit = 0
        self.move_cost = 1
        self.gamma = 0.5
        self.probabilistic_action = {"Up": 0.8, "Right": 0.8, "Left": 1, "Down": 1}

    def where_is(self):
        """Returns the current location of the agent and possible reachable states"""
        self.current_location.display()
        print("Profit of the agent so far: {}".format(self.profit))
        # print("\n{}".format(Agent.__doc__))

    def has_reached_goal(self):
        if self.current_location.goal_state_check():
            print("Hurray! Detective found the treasure!! :D :D")
        else:
            print("Not yet dude! Just a bit more exploration would do the magic..")

    def perform_value_iteration(self):
        self.current_location.value_iteration(self.gamma)

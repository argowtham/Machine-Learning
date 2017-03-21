import numpy as np


class GridSquare(object):
    """A single square in the grid world. It has the following properties

    Attributes:
        x: indicates the x co-ordinate of the cell
        """

    def __init__(self, row, col, world):
        self.x = row
        self.y = col
        self.location = [row, col]
        self.next_state = {"up": self, "down": self, "right": self, "left": self}
        self.world = world
        self.value = 0
        self.reward = None
        self.find_reward()

    def find_next_states(self):
        # TODO Find next states from current states
        pass

    def find_reward(self):
        if self.location == self.world.pit:
            self.reward = -50
        elif self.location == self.world.goal:
            self.reward = 10
        elif self.location in self.world.unreachable_states:
            self.reward = np.nan
        else:
            self.reward = 0

    def value_iteration(self, gamma):
        max_value = -float('inf')
        for action, state in self.next_state.items():
            if state.value >= max_value:
                max_value = state.value
        self.value = self.reward + gamma * max_value

    def find_adjacent_states(self):
        next_row = self.x + 1
        next_col = self.y + 1
        prev_row = self.x - 1
        prev_col = self.y - 1

        unreachable_states = self.world.unreachable_states
        if next_row < self.world.nrows and [next_row, self.y] not in unreachable_states:
            self.next_state["up"] = self.world.world_grid[next_row][self.y]

        if prev_row > 0 and [prev_row, self.y] not in unreachable_states:
            self.next_state["down"] = self.world.world_grid[prev_row][self.y]

        if next_col < self.world.ncols and [self.x, next_col] not in unreachable_states:
            self.next_state["right"] = self.world.world_grid[self.x][next_col]

        if prev_col > 0 and [self.x, prev_col] not in unreachable_states:
            self.next_state["left"] = self.world.world_grid[self.x][prev_col]

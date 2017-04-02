from Reinforcement_Learning.GridSquare import GridSquare
import sys


class GridWorld(object):
    """Provides a blue print for the grid world domain. It has the following properties:

    Attributes:
        :nrows An integer representing the number of rows
        :ncols An integer representing the number of columns
        :n An integer representing the number of possible cells"""

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.n = self.ncols * self.nrows
        self.world_grid = []
        self.unreachable_states = [[1, 1], [1, 3]]
        self.pit = [3, 1]
        self.goal = [4, 3]
        self.create_grid_squares()
        self.find_next_states()

    def create_grid_squares(self):
        for i in range(self.nrows):
            row_squares = []
            for j in range(self.ncols):
                row_squares.append(GridSquare(i, j, self))
            self.world_grid.append(row_squares)

    def find_next_states(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                state = self.world_grid[i][j]
                if state not in self.unreachable_states:
                    state.find_adjacent_states()

    def display_world(self, param):
        if param == "world":
            for i in range(self.nrows):
                for j in range(self.ncols):
                    sys.stdout.write(str(self.world_grid[i][j].location) + " ")
                print()
            print()
        elif param == "value":
            for i in range(self.nrows):
                for j in range(self.ncols):
                    sys.stdout.write(str(self.world_grid[i][j].value) + " ")
                print()
            print()
        elif param == "reward":
            for i in range(self.nrows):
                for j in range(self.ncols):
                    sys.stdout.write(str(self.world_grid[i][j].reward) + " ")
                print()
            print()
        elif param == "q_value":
            for i in range(self.nrows):
                for j in range(self.ncols):
                    sys.stdout.write("Q_values of state {} is {}".format([i, j], self.world_grid[i][j].q_value))
                    print()

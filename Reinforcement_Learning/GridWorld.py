class GridWorld(object):
    """Provides a blue print for the grid world domain. It has the following properties:

    Attributes:
        nrows: An integer representing the number of rows
        ncols: An integer representing the number of columns
        n: An integer representing the number of possible cells"""

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.n = self.ncols * self.nrows

if __name__ == "__main__":
    pass
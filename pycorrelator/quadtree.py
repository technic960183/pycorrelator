import numpy as np

class QuadTree:

    def __init__(self):
        self.root = None

    def insert(self, x, y):
        if self.root is None:
            self.root = _Node(x, y)
            return True
        else:
            return self.root.insert(x, y)

    def query(self, x, y):
        if self.root is None:
            return False
        else:
            return self.root.query(x, y)


class _Node:
    '''
    Private class Node. User should not be able to access this class directly.
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.quads = [None, None, None, None]

    def insert(self, x, y):
        if self.x == x and self.y == y:
            return False  # Duplicate found
        quad = self.get_quadrant(x, y)
        if quad is None:
            self.set_quadrant(x, y, _Node(x, y))
            return True
        else:
            return quad.insert(x, y)

    def query(self, x, y):
        if self.x == x and self.y == y:
            return True
        quad = self.get_quadrant(x, y)
        if quad is None:
            return False
        return quad.query(x, y)

    def set_quadrant(self, x, y, node):
        self.quads[self.index(x, y)] = node
        return node

    def get_quadrant(self, x, y):
        return self.quads[self.index(x, y)]

    def index(self, x, y):
        dr = 1 if x > self.x else 0
        dd = 1 if y > self.y else 0
        return dr * 2 + dd


class DataQuadTree(QuadTree):
    def __init__(self):
        self.root = None

    def insert(self, x, y, data):
        if self.root is None:
            self.root = DataNode(x, y, data)
        else:
            self.root.insert(x, y, data)

    def query(self, x, y):
        if self.root is None:
            return list()
        return self.root.query(x, y)
    
    def query_batch(self, x_nparray, y_nparray):
        if self.root is None:
            return list()
        return self.root.query_batch(x_nparray, y_nparray)

    def length(self, x, y):
        if self.root is None:
            return 0
        return len(self.root.query(x, y))


class DataNode(_Node):
    def __init__(self, x, y, data):
        super().__init__(x, y)
        self.list_data = [data]

    def insert(self, x, y, data):
        if (x, y) == (self.x, self.y):
            self.list_data.append(data)
        else:
            quad = self.get_quadrant(x, y)
            if quad is None:
                self.set_quadrant(x, y, DataNode(x, y, data))
            else:
                quad.insert(x, y, data)

    def query(self, x, y):
        if (x, y) == (self.x, self.y):
            return self.list_data
        quad = self.get_quadrant(x, y)
        if quad is None:
            return list()
        return quad.query(x, y)
    
    def query_batch(self, x_nparray, y_nparray): # Slow DONT USE
        '''
        CAUTION: This method is slow. Do not use it.
        '''
        if len(x_nparray) == 0:
            return list()
        rtn = []
        x_diff = x_nparray - self.x
        if np.all(x_diff > 0) or np.all(x_diff < 0):
            y_diff = y_nparray - self.y
            if np.all(y_diff > 0) or np.all(y_diff < 0):
                quad = self.get_quadrant(x_nparray[0], y_nparray[0])
                if quad is None:
                    return list()
                return quad.query_batch(x_nparray, y_nparray)
        for x, y in zip(x_nparray, y_nparray):
            rtn += self.query(x, y)
        return rtn

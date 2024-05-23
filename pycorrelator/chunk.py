import numpy as np
from numpy.typing import NDArray


class Chunk:

    def __init__(self, chunk_id, ra, dec, discription=None):
        self.chunk_id = chunk_id
        self.discription = discription if discription != None else f"Chunk {chunk_id} ({ra:3f}, {dec:3f})"
        self.central_data = np.empty((0, 2), dtype=np.float64)
        self.boundary_data = np.empty((0, 2), dtype=np.float64)
        self.central_index = np.empty((0), dtype=np.int64)
        self.boundary_index = np.empty((0), dtype=np.int64)
        self.chunk_ra = ra
        self.chunk_dec = dec
        self.max_size = None

    def add_central_data(self, data, index):
        self.central_data = np.concatenate([self.central_data, data])
        self.central_index = np.concatenate([self.central_index, index])

    def add_boundary_data(self, data, index):
        self.boundary_data = np.concatenate([self.boundary_data, data])
        self.boundary_index = np.concatenate([self.boundary_index, index])

    def get_data(self) -> NDArray[np.float64]:
        return np.concatenate([self.central_data, self.boundary_data])

    def get_index(self) -> NDArray[np.int64]:
        return np.concatenate([self.central_index, self.boundary_index])

    def get_center(self):
        return self.chunk_ra, self.chunk_dec

    def farest_distance(self, distance=None):
        if distance == None:
            return self.max_size
        self.max_size = distance

    def __len__(self):
        return len(self.get_index())

    def __repr__(self):
        return f"Chunk {self.chunk_id} ({self.chunk_ra:.1f}, {self.chunk_dec:.1f}): {len(self)} objects"

    def __str__(self):
        return self.discription + f" with {len(self)} objects"

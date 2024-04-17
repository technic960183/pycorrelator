import pandas as pd


class Chunk:

    def __init__(self, chunk_id, ra, dec, discription=None):
        self.chunk_id = chunk_id
        self.discription = discription if discription != None else f"Chunk {chunk_id} ({ra:3f}, {dec:3f})"
        self.central_data = pd.DataFrame()
        self.boundary_data = pd.DataFrame()
        self.chunk_ra = ra
        self.chunk_dec = dec
        self.max_size = None

    def add_central_data(self, data):
        self.central_data = pd.concat([self.central_data, data])

    def add_boundary_data(self, data):
        self.boundary_data = pd.concat([self.boundary_data, data])

    def get_data(self) -> pd.DataFrame:
        return pd.concat([self.central_data, self.boundary_data])

    def get_center(self):
        return self.chunk_ra, self.chunk_dec

    def farest_distance(self, distance=None):
        if distance == None:
            return self.max_size
        self.max_size = distance

    def __len__(self):
        return len(self.get_data())

    def __repr__(self):
        return f"Chunk {self.chunk_id} ({self.chunk_ra:.1f}, {self.chunk_dec:.1f}): {len(self)} objects"

    def __str__(self):
        return self.discription + f" with {len(self)} objects"

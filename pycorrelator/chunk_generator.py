from .catalog import Catalog
from .chunk import Chunk


class ChunkGenerator:

    def __init__(self, margin):
        '''
        Purpose: Initialize the chunk generator
        '''
        self.chunks: list[Chunk] = []
        self.margin = margin

    def get_chunk(self, chunk_id):
        return self.chunks[chunk_id]

    def distribute(self, catalog: Catalog):
        '''
        Purpose: Distribute the data into chunks
        Parameters:
        - catalog: The catalog to be distributed
        Returns:
        - chunks: list of chunks with data
        '''
        coordiantes = catalog.get_coordiantes()
        indexes = catalog.get_indexes()
        ra, dec = coordiantes[:, 0], coordiantes[:, 1]

        # Get chunk ids for central coordinates
        central_chunk_ids = self.coor2id_central(ra, dec)
        for i in range(len(self.chunks)):
            mask = (central_chunk_ids == i)
            self.chunks[i].add_central_data(coordiantes[mask], indexes[mask])

        # Get chunk ids for boundary coordinates
        boundary_chunk_indices = self.coor2id_boundary(ra, dec)
        for boundary_chunk_id, indices in enumerate(boundary_chunk_indices):  # May be a bug here
            self.chunks[boundary_chunk_id].add_boundary_data(coordiantes[indices], indexes[indices])

        return self.chunks

    def generate(self):
        '''
        --- SHOULD BE overridden by subclass ---
        Purpose: Generate the chunks
        '''
        raise NotImplementedError()

    def coor2id_central(self, ra, dec):
        '''
        --- SHOULD BE overridden by subclass ---
        Purpose: Tell which chunk the given coordinate belongs to (how to divide the sky)
        Parameters:
        - ra: numpy array of RA of size n
        - dec: numpy array of Dec of size n
        Returns:
        - chink_id: numpy array of chunk_id of size n
        '''
        raise NotImplementedError()

    def coor2id_boundary(self, ra, dec):
        '''
        --- SHOULD BE overridden by subclass ---
        Purpose: Tell which boundary of chunk the given coordinate assigned to (how to divide the sky)
                 If the object is located within the 2 * margin with the boundary of a chunk,
                 it should be contained in the list of object indexes of that chunk. 
        Parameters:
        - ra: numpy array of RA
        - dec: numpy array of Dec
        Returns:
        - list_of_chunk_of_list_of_object_index: a list of lists contain the index of the object 
        '''
        raise NotImplementedError()

import numpy as np
import pandas as pd
from .toolbox_spherical import great_circle_distance


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


class ChunkGenerator:
    #### MOVE TO ChunkGenerator.py ####
    #### DELETE LATER ####

    def __init__(self, margin):
        '''
        Purpose: Initialize the chunk generator
        '''
        self.chunks = []
        self.margin = margin

    def get_chunk(self, chunk_id):
        return self.chunks[chunk_id]

    def distribute(self, data: pd.DataFrame):
        '''
        Purpose: Distribute the data into chunks
        Parameters:
        - data: pandas dataframe of data (should contain columns "Ra" and "Dec")
        Returns:
        - chunks: list of chunks
        '''
        ra = data["Ra"].values  # Could be optimized
        dec = data["Dec"].values  # Could be optimized

        # Get chunk ids for central coordinates
        central_chunk_ids = self.coor2id_central(ra, dec)
        for i in range(len(self.chunks)):
            mask = (central_chunk_ids == i)
            self.chunks[i].add_central_data(data[mask])

        # Get chunk ids for boundary coordinates
        boundary_chunk_indices = self.coor2id_boundary(ra, dec)
        for boundary_chunk_id, indices in enumerate(boundary_chunk_indices):  # May be a bug here
            self.chunks[boundary_chunk_id].add_boundary_data(data.iloc[indices])

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


class ChunkGeneratorByGrid(ChunkGenerator):
    """
    This class is responsible for generating chunks based on a grid structure.

    Attributes:
        grid : list of `Chunk` objects
            - The grid structure is designed to cover the celestial sphere.

            Polar Chunks:
            - Polar Chunk North: This chunk is responsible for capturing the northernmost celestial sphere 
              specifically for declinations (Dec) greater than 60°.
            - Polar Chunk South: This chunk is responsible for capturing the southernmost celestial sphere 
              specifically for declinations (Dec) less than -60°.

            Middle Chunks:
            - The middle chunks cover the declinations between 0° and 60° (both inclusive) for the northern 
              hemisphere and between 0° and -60° (both inclusive) for the southern hemisphere.
            - The central declinations for these bands are 30° and -30°.
            - Each declination band (e.g., 30° and -30°) is further divided based on right ascension (RA) values 
              into six chunks. These chunks have central RA values of [30°, 90°, 150°, 210°, 270°, 330°].
            - The combination of these declinations and RAs creates the grid of middle chunks.

    Methods:
        generate:
            Generates the chunks based on the grid structure.
    """

    #### DEPRECATED CLASS ####
    #### DELETE LATER ####

    def __init__(self, margin):
        super().__init__(margin=margin)
        self.generate()

    def generate(self):
        # Polar chunks
        cn = Chunk(0, 180, 90, "Polar Chunk North")  # For Dec > 60°
        cn.farest_distance(distance=90-60+2*self.margin)
        cs = Chunk(1, 180, -90, "Polar Chunk South")  # For Dec < -60°
        cs.farest_distance(distance=90-60+2*self.margin)
        self.chunks += [cn, cs]

        # Middle chunks (0° ≤ Dec ≤ 60° and 0° ≥ Dec ≥ -60°)
        chunk_id = 2
        size = great_circle_distance(0, 0, 30+2*self.margin, 30+2*self.margin)
        for dec in [30, -30]:  # Central declination for the two bands
            for ra in [30, 90, 150, 210, 270, 330]:  # Central right ascension values
                c = Chunk(chunk_id, ra, dec)
                c.farest_distance(distance=size)
                self.chunks.append(c)
                chunk_id += 1

    def coor2id_central(self, ra, dec):
        chunk_ids = np.zeros_like(ra, dtype=int)

        # Polar chunks
        chunk_ids[dec > 60] = 0
        chunk_ids[dec < -60] = 1

        # Middle chunks
        for i in range(2, 14):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(ra - chunk_ra)
            dec_diff = np.abs(dec - chunk_dec)
            mask = ((ra_diff <= 30) & (dec_diff <= 30))
            chunk_ids[mask] = i

        return chunk_ids

    def coor2id_boundary(self, ra, dec):
        margin = self.margin
        list_of_chunk_of_list_of_object_index = []

        # Polar chunks
        north_polar_chunk = (dec < 60) & (dec >= 60 - 2 * margin)
        south_polar_chunk = (dec > -60) & (dec <= -60 + 2 * margin)

        # Append the indices of objects that belong to the polar chunk boundaries
        list_of_chunk_of_list_of_object_index.append(list(np.where(north_polar_chunk)[0]))
        list_of_chunk_of_list_of_object_index.append(list(np.where(south_polar_chunk)[0]))

        # Middle chunks
        for i in range(2, 14):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(chunk_ra - ra)
            ra_diff = np.minimum(ra_diff, 360 - ra_diff)
            dec_diff = np.abs(dec - chunk_dec)
            mask_ra = (ra_diff >= 30) & (ra_diff <= 30 + 2 * margin) & (dec_diff <= 30 + 2 * margin)
            mask_dec = (dec_diff >= 30) & (dec_diff <= 30 + 2 * margin) & (ra_diff <= 30 + 2 * margin)
            mask = mask_ra | mask_dec
            list_of_chunk_of_list_of_object_index.append(list(np.where(mask)[0]))

        return list_of_chunk_of_list_of_object_index


class ChunkGeneratorByDenseGrid(ChunkGenerator):
    #### DEPRECATED CLASS ####
    #### DELETE LATER ####

    def __init__(self, margin):
        super().__init__(margin=margin)
        self.generate()

    def generate(self):
        # Ultra-Polar Chunk North-1 and South-1
        cn = Chunk(0, 180, 90, "Ultra-Polar Chunk North-1")  # For Dec > 75°
        cn.farest_distance(distance=90-75+2*self.margin)
        cs = Chunk(1, 180, -90, "Ultra-Polar Chunk South-1")  # For Dec < -75°
        cs.farest_distance(distance=90-75+2*self.margin)
        self.chunks += [cn, cs]

        # RA-divided Ultra-Polar Chunk North-2 and South-2
        chunk_id = 2
        for dec in [67.5, -67.5]:  # Center of the dec bands
            for ra in [30, 90, 150, 210, 270, 330]:  # Central RA values
                c = Chunk(chunk_id, ra, dec)
                d1 = great_circle_distance(30, 67.5, 60+2*self.margin, 75+2*self.margin)
                d2 = great_circle_distance(30, 67.5, 60+2*self.margin, 60-2*self.margin)
                c.farest_distance(distance=max(d1, d2))
                self.chunks.append(c)
                chunk_id += 1

        # Dense Middle Chunks
        for dec in [45, 15, -15, -45]:  # Central declination values
            for ra in [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]:  # Central RA values
                c = Chunk(chunk_id, ra, dec)
                d1 = great_circle_distance(15, abs(dec), 30+2*self.margin, abs(dec)+15+2*self.margin)
                d2 = great_circle_distance(15, abs(dec), 30+2*self.margin, abs(dec)-15-2*self.margin)
                c.farest_distance(distance=max(d1, d2))
                self.chunks.append(c)
                chunk_id += 1

    def coor2id_central(self, ra, dec):
        chunk_ids = np.zeros_like(ra, dtype=int)

        # Ultra-Polar chunks
        chunk_ids[dec >= 75] = 0
        chunk_ids[dec <= -75] = 1

        # RA-divided Ultra-Polar chunks
        for i in range(2, 14):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(ra - chunk_ra)
            dec_diff = np.abs(dec - chunk_dec)
            mask = ((ra_diff <= 30) & (dec_diff <= 7.5))
            chunk_ids[mask] = i

        # RA-divided Dense Middle Chunks
        for i in range(14, len(self.chunks)):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(ra - chunk_ra)
            dec_diff = np.abs(dec - chunk_dec)
            mask = ((ra_diff <= 15) & (dec_diff <= 15))
            chunk_ids[mask] = i

        return chunk_ids

    def coor2id_boundary(self, ra, dec):
        margin = self.margin
        list_of_chunk_of_list_of_object_index = []

        # Ultra-Polar Chunk boundaries
        north_polar_chunk_1 = (dec < 75) & (dec >= 75 - 2 * margin)
        south_polar_chunk_1 = (dec > -75) & (dec <= -75 + 2 * margin)
        list_of_chunk_of_list_of_object_index.append(list(np.where(north_polar_chunk_1)[0]))
        list_of_chunk_of_list_of_object_index.append(list(np.where(south_polar_chunk_1)[0]))

        # RA-divided Ultra-Polar Chunk boundaries
        for i in range(2, 14):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(chunk_ra - ra)
            ra_diff = np.minimum(ra_diff, 360 - ra_diff)
            dec_diff = np.abs(dec - chunk_dec)
            mask_ra = (ra_diff >= 30) & (ra_diff <= 30 + 2 * margin) & (dec_diff <= 7.5 + 2 * margin)
            mask_dec = (dec_diff >= 7.5) & (dec_diff <= 7.5 + 2 * margin) & (ra_diff <= 30 + 2 * margin)
            mask = mask_ra | mask_dec
            list_of_chunk_of_list_of_object_index.append(list(np.where(mask)[0]))

        # RA-divided Dense Middle Chunk boundaries
        for i in range(14, len(self.chunks)):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(chunk_ra - ra)
            ra_diff = np.minimum(ra_diff, 360 - ra_diff)
            dec_diff = np.abs(dec - chunk_dec)
            mask_ra = (ra_diff >= 15) & (ra_diff <= 15 + 2 * margin) & (dec_diff <= 15 + 2 * margin)
            mask_dec = (dec_diff >= 15) & (dec_diff <= 15 + 2 * margin) & (ra_diff <= 15 + 2 * margin)
            mask = mask_ra | mask_dec
            list_of_chunk_of_list_of_object_index.append(list(np.where(mask)[0]))

        return list_of_chunk_of_list_of_object_index


class ChunkGeneratorBySuperDenseGrid(ChunkGenerator):
    #### DEPRECATED CLASS ####
    #### DELETE LATER ####

    def __init__(self, margin):
        super().__init__(margin=margin)
        self.generate()

    def generate(self):
        # Ultra-Polar Chunk North-1 and South-1
        self.chunks.append(Chunk(0, 180, 90, "Ultra-Polar Chunk North-1"))  # For Dec > 80°
        self.chunks.append(Chunk(1, 180, -90, "Ultra-Polar Chunk South-1"))  # For Dec < -80°

        # RA-divided Ultra-Polar Chunk North-2 and South-2
        chunk_id = 2
        for dec in [70, -70]:  # Center of the dec bands
            for ra in [7.5 + 15 * i for i in range(24)]:  # Central RA values
                self.chunks.append(Chunk(chunk_id, ra, dec))
                chunk_id += 1

        # Dense Middle Chunks
        for dec in [60 - 7.5 - 15 * i for i in range(8)]:  # Central declination values
            for ra in [7.5 + 15 * i for i in range(24)]:  # Central RA values
                self.chunks.append(Chunk(chunk_id, ra, dec))
                chunk_id += 1

    def coor2id_central(self, ra, dec):
        chunk_ids = np.zeros_like(ra, dtype=int)

        # Ultra-Polar chunks
        chunk_ids[dec >= 80] = 0
        chunk_ids[dec <= -80] = 1

        # RA-divided Ultra-Polar chunks
        for i in range(2, 2 + 12 * 4):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(ra - chunk_ra)
            dec_diff = np.abs(dec - chunk_dec)
            mask = ((ra_diff <= 7.5) & (dec_diff <= 10))
            chunk_ids[mask] = i

        # RA-divided Dense Middle Chunks
        for i in range(2 + 12 * 4, len(self.chunks)):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(ra - chunk_ra)
            dec_diff = np.abs(dec - chunk_dec)
            mask = ((ra_diff <= 7.5) & (dec_diff <= 7.5))
            chunk_ids[mask] = i

        return chunk_ids

    def coor2id_boundary(self, ra, dec):
        margin = self.margin
        list_of_chunk_of_list_of_object_index = []

        # Ultra-Polar Chunk boundaries
        north_polar_chunk_1 = (dec < 80) & (dec >= 80 - 2 * margin)
        south_polar_chunk_1 = (dec > -80) & (dec <= -80 + 2 * margin)
        list_of_chunk_of_list_of_object_index.append(list(np.where(north_polar_chunk_1)[0]))
        list_of_chunk_of_list_of_object_index.append(list(np.where(south_polar_chunk_1)[0]))

        # RA-divided Ultra-Polar Chunk boundaries
        for i in range(2, 2 + 12 * 4):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(chunk_ra - ra)
            ra_diff = np.minimum(ra_diff, 360 - ra_diff)
            dec_diff = np.abs(dec - chunk_dec)
            mask_ra = (ra_diff >= 7.5) & (ra_diff <= 7.5 + 2 * margin) & (dec_diff <= 10 + 2 * margin)
            mask_dec = (dec_diff >= 10) & (dec_diff <= 10 + 2 * margin) & (ra_diff <= 7.5 + 2 * margin)
            mask = mask_ra | mask_dec
            list_of_chunk_of_list_of_object_index.append(list(np.where(mask)[0]))

        # RA-divided Dense Middle Chunk boundaries
        for i in range(2 + 12 * 4, len(self.chunks)):
            chunk_ra = self.chunks[i].chunk_ra
            chunk_dec = self.chunks[i].chunk_dec
            ra_diff = np.abs(chunk_ra - ra)
            ra_diff = np.minimum(ra_diff, 360 - ra_diff)
            dec_diff = np.abs(dec - chunk_dec)
            mask_ra = (ra_diff >= 7.5) & (ra_diff <= 7.5 + 2 * margin) & (dec_diff <= 7.5 + 2 * margin)
            mask_dec = (dec_diff >= 7.5) & (dec_diff <= 7.5 + 2 * margin) & (ra_diff <= 7.5 + 2 * margin)
            mask = mask_ra | mask_dec
            list_of_chunk_of_list_of_object_index.append(list(np.where(mask)[0]))

        return list_of_chunk_of_list_of_object_index

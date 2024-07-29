from typing import Optional
import numpy as np
from .chunk import Chunk
from .chunk_generator import ChunkGenerator
from .utilities_spherical import great_circle_distance


class GridChunkConfig:

    def __init__(self, center, margin, width: Optional[tuple] = None, dec_bound: Optional[float] = None):
        '''
        Parameters
        ----------

        center : tuple
            Tuple of two floats (ra, dec) in degrees.
        margin : float
            Margin in degrees.
        width : tuple, optional
            Tuple of two floats (w_ra, w_dec) in degrees (for ring chunks).
        dec_bound : float, optional
            Float in degrees (for polar chunks).

        Note
        ----
            Specify either the width for ring chunks or the dec_bound for polar chunks. 
        '''
        self.margin = margin
        if dec_bound is not None and width is not None:
            raise ValueError("You must specify either the width or the dec_bound, not both")
        self.config = {}
        self.config['center_ra'] = center[0]
        self.config['center_dec'] = center[1]
        if dec_bound is not None:
            if center[1] == 90 or center[1] == -90:
                if dec_bound <= 0:
                    raise ValueError("The dec_bound must be positive")
                self.config['chunk_type'] = 'polar'
                self.config['dec_bound'] = dec_bound
            else:
                raise ValueError("The center_dec must be either 90 or -90 for polar chunks")
        elif width is not None:
            if width[0] <= 0 or width[1] <= 0:
                raise ValueError("The width must be positive")
            self.config['chunk_type'] = 'ring'
            self.config['delta_ra'] = width[0] / 2
            self.config['delta_dec'] = width[1] / 2
        else:
            raise ValueError("You must specify either the width or the dec_bound")

    def __getitem__(self, key):
        return self.config[key]

    def get_max_radius(self):
        if self.config['chunk_type'] == 'polar':
            return 90 - self.config['dec_bound'] + self.margin
        elif self.config['chunk_type'] == 'ring':
            ra, dec = self.config['center_ra'], self.config['center_dec']
            d_ra, d_dec = self.config['delta_ra'], self.config['delta_dec']
            m = self.margin
            d1 = great_circle_distance(ra, dec, ra + d_ra + m, dec + d_dec + m)
            d2 = great_circle_distance(ra, dec, ra + d_ra + m, dec - d_dec - m)
            return max(d1, d2)
        else:
            raise ValueError(f"Unknown chunk type: {self.config['chunk_type']}")


class GridChunkGenerator(ChunkGenerator):
    def __init__(self, margin):
        super().__init__(margin=margin)
        self.config_polar = []
        self.config_ring = []

    def add_polar_config(self, center, dec_bound):
        self.config_polar.append(GridChunkConfig(center, self.margin, dec_bound=dec_bound))

    def add_ring_config(self, center, width):
        self.config_ring.append(GridChunkConfig(center, self.margin, width=width))

    def get_all_config(self):
        return self.config_polar + self.config_ring

    def set_symmetric_ring_chunk(self, polar_dec, Ns_horizontal_ring):
        # Polar chunks
        if polar_dec <= 0:
            raise ValueError("The polar_dec must be positive")
        self.add_polar_config((180, 90), polar_dec)
        self.add_polar_config((180, -90), polar_dec)
        # Ring chunks
        N_dec = len(Ns_horizontal_ring)
        width_dec = 2 * polar_dec / N_dec
        for i in range(N_dec):  # Iterate over declination bands
            pos_dec = polar_dec - (2 * polar_dec) / (2 * N_dec) * (1 + 2 * i)
            N_Ra = Ns_horizontal_ring[i]
            width_ra = 360 / N_Ra
            for j in range(N_Ra):  # Iterate over right ascension values
                pos_ra = width_ra / 2 + width_ra * j
                self.add_ring_config((pos_ra, pos_dec), (width_ra, width_dec))
        self.generate()

    def generate(self):
        if len(self.config_polar) != 2:
            raise ValueError("The number of polar chunks must be 2.")
        chunk_id = 0
        for config in self.get_all_config():
            chunk = Chunk(chunk_id, config['center_ra'], config['center_dec'])
            chunk.farest_distance(distance=config.get_max_radius())
            self.chunks.append(chunk)
            chunk_id += 1

    def coor2id_central(self, ra, dec):
        chunk_ids = np.zeros_like(ra, dtype=int)
        # Polar chunks
        chunk_ids[dec > self.config_polar[0]['dec_bound']] = 0
        chunk_ids[dec < -self.config_polar[1]['dec_bound']] = 1
        # Ring chunks
        for i, config in enumerate(self.config_ring):
            ra_diff = np.abs(ra - config['center_ra'])
            ra_diff = np.minimum(ra_diff, 360 - ra_diff) # Not necessary. The central parts don't cross the 0-360 boundary.
            dec_diff = np.abs(dec - config['center_dec'])
            mask_ra = (ra_diff <= config['delta_ra'])
            mask_dec = (dec_diff <= config['delta_dec'])
            mask = mask_ra & mask_dec
            chunk_ids[mask] = i + 2

        return chunk_ids

    def coor2id_boundary(self, ra, dec):
        margin = self.margin
        list_of_chunk_of_list_of_object_index = []

        # Polar chunks
        N_polar_bound = self.config_polar[0]['dec_bound']
        S_polar_bound = -self.config_polar[1]['dec_bound']
        north_polar_chunk = (dec < N_polar_bound) & (dec >= N_polar_bound - margin)
        south_polar_chunk = (dec > S_polar_bound) & (dec <= S_polar_bound + margin)

        # Append the indices of objects that belong to the polar chunk boundaries
        list_of_chunk_of_list_of_object_index.append(list(np.where(north_polar_chunk)[0]))
        list_of_chunk_of_list_of_object_index.append(list(np.where(south_polar_chunk)[0]))

        # Middle chunks
        for config in self.config_ring:
            ra_diff = np.abs(ra - config['center_ra'])
            ra_diff = np.minimum(ra_diff, 360 - ra_diff) # Necessary. The boundary parts DO cross the 0-360 boundary.
            dec_diff = np.abs(dec - config['center_dec'])
            mask_ra = (ra_diff >= config['delta_ra']) & (ra_diff <= config['delta_ra'] + margin) & (
                dec_diff <= config['delta_dec'] + margin)
            mask_dec = (dec_diff >= config['delta_dec']) & (dec_diff <= config['delta_dec'] + margin) & (
                ra_diff <= config['delta_ra'] + margin)
            mask = mask_ra | mask_dec
            list_of_chunk_of_list_of_object_index.append(list(np.where(mask)[0]))

        return list_of_chunk_of_list_of_object_index


class ChunkGeneratorByGrid(GridChunkGenerator):
    def __init__(self, margin):
        super().__init__(margin=margin)
        self.set_symmetric_ring_chunk(polar_dec=60, Ns_horizontal_ring=[6, 6])


class ChunkGeneratorByDenseGrid(GridChunkGenerator):
    def __init__(self, margin):
        super().__init__(margin=margin)
        self.set_symmetric_ring_chunk(polar_dec=75, Ns_horizontal_ring=[6, 12, 12, 6])


class ChunkGeneratorBySuperDenseGrid(GridChunkGenerator):
    def __init__(self, margin):
        super().__init__(margin=margin)
        self.set_symmetric_ring_chunk(polar_dec=80, Ns_horizontal_ring=[24]*10)

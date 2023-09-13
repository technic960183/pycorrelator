import unittest
import numpy as np
from SphericalMatch.Chunk import ChunkGeneratorByGrid


class TestChunkGeneratorByGrid_coor2id_central(unittest.TestCase):

    def setUp(self):
        self.chunk_gen = ChunkGeneratorByGrid()

    def test_polar_regions(self):
        ra = np.array([0, 180, 0, 180])
        dec = np.array([90, 85, -90, -85])
        expected_chunk_ids = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(self.chunk_gen.coor2id_central(ra, dec), expected_chunk_ids)

    def test_middle_chunks_north(self):
        ra = np.array([15, 75, 135, 195, 255, 315, 45])
        dec = np.array([30, 30, 30, 30, 30, 30, 5])
        expected_chunk_ids = np.array([2, 3, 4, 5, 6, 7, 2])
        np.testing.assert_array_equal(self.chunk_gen.coor2id_central(ra, dec), expected_chunk_ids)

    def test_middle_chunks_south(self):
        ra = np.array([15, 75, 135, 195, 255, 315, 45])
        dec = np.array([-30, -30, -30, -30, -30, -30, -5])
        expected_chunk_ids = np.array([8, 9, 10, 11, 12, 13, 8])
        np.testing.assert_array_equal(self.chunk_gen.coor2id_central(ra, dec), expected_chunk_ids)

    def test_boundaries(self):
        ra = np.array([60, 120, 180, 240, 300, 360]) - 0.1
        dec = np.array([59.9, 0.1, -59.9, 0.1, 59.9, 0.1])
        # Assuming a point on the boundary belongs to the chunk to its right (or above for Dec)
        expected_chunk_ids = np.array([2, 3, 10, 5, 6, 7])
        np.testing.assert_array_equal(self.chunk_gen.coor2id_central(ra, dec), expected_chunk_ids)


class TestChunkGeneratorByGrid_coor2id_boundary(unittest.TestCase):

    def setUp(self):
        '''Setup the ChunkGeneratorByGrid instance for testing.'''
        self.chunk_gen = ChunkGeneratorByGrid()

    def test_polar_chunks_boundaries(self):
        ra = np.array([15, 15])
        dec = np.array([58, -58])  # Near the polar chunk boundaries
        tolerance = 2.5
        expected_result = [[0], [1], [], [], [], [], [], [], [], [], [], [], [], []]
        result = self.chunk_gen.coor2id_boundary(ra, dec, tolerance)
        self.assertEqual(result, expected_result)

    def test_middle_chunks_boundaries(self):
        ra = np.array([0, 60, 120, 180, 240, 300]) + 0.01
        dec = np.array([28, 28, 28, 28, 28, 28])  # Near the middle chunk boundaries
        tolerance = 1
        expected_result = [[], [], [1], [2], [3], [4], [5], [0], [], [], [], [], [], []]
        result = self.chunk_gen.coor2id_boundary(ra, dec, tolerance)
        self.assertEqual(result, expected_result)

    def test_objects_outside_tolerance_boundary(self):
        ra = np.array([4])
        dec = np.array([56])  # Outside the tolerance for polar chunk boundaries
        tolerance = 1.5
        expected_result = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
        result = self.chunk_gen.coor2id_boundary(ra, dec, tolerance)
        self.assertEqual(result, expected_result)


# Running the tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestChunkGeneratorByGrid_coor2id_boundary)
unittest.TextTestRunner(verbosity=2).run(suite)

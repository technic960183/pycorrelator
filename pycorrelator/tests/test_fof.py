import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import numpy as np
from numpy.typing import NDArray
from pycorrelator import point_offset, generate_random_point
# from pycorrelator import group_by_disjoint_set, group_by_DFS
from pycorrelator import fof


def generate_celestial_grid(**kwargs) -> list[tuple[float, float]]:
    """Generate a grid on the celestial sphere with specified step sizes and bounds.

    Keyword Arguments
    -----------------
    ra_step: float
        Step size for Right Ascension (default: 10).
    dec_step: float
        Step size for Declination (default: 10).
    ra_offset: float
        Offset for RA (default: 0).
    dec_offset: float
        Offset for DEC (default: 0).
    ra_bounds: tuple
        A tuple containing the lower and upper bounds for RA (default: (0, 360)).
    dec_bounds: tuple | float
        A tuple containing the lower and upper bounds for DEC (default: (-60, 60)).
        Can also be an absolute value which will be interpreted as (value, -value).

    Returns
    -------
    list[tuple[float, float]]
        List of (RA, Dec) coordinates for the grid points.
    """

    ra_step = kwargs.get('ra_step', 10)
    dec_step = kwargs.get('dec_step', 10)
    ra_offset = kwargs.get('ra_offset', 0)
    dec_offset = kwargs.get('dec_offset', 0)
    ra_bounds = kwargs.get('ra_bounds', (0, 360))
    dec_bounds = kwargs.get('dec_bounds', 60)

    # Adjust bounds if they are provided as absolute values
    if isinstance(ra_bounds, (int, float)):
        raise ValueError("ra_bounds must be a tuple")
    if isinstance(dec_bounds, (int, float)):
        dec_bounds = (-abs(dec_bounds), abs(dec_bounds))

    ras = np.arange(ra_bounds[0] + ra_offset, ra_bounds[1], ra_step)
    if ra_bounds[1] != 360 and ras[-1] + ra_step == ra_bounds[1]:
        ras = np.append(ras, ra_bounds[1])
    decs = np.arange(dec_bounds[0] + dec_offset, dec_bounds[1], dec_step)
    if decs[-1] + dec_step == dec_bounds[1]:
        decs = np.append(decs, dec_bounds[1])
    grid = [(ra, dec) for ra in ras for dec in decs]
    return grid


def create_groups_from_grid(grid: list[tuple[float, float]], 
                            tolerance=1, seed=None, fraction=0.5, 
                            ring_radius=(0, 1)) -> tuple[list[list[tuple[float, float]]], NDArray]:
    """
    Randomly pick half of the grid points and create groups around them.
    For each selected grid point, use the point_offset() function to create several points 
    within a tolerance (default 1 degree) circle around the central point.
    Returns a list of groups, where each group is a list of (RA, Dec) coordinates.
    """
    if seed is not None:
        seed = np.random.randint(0, 1e6)
    np.random.seed(seed)
    np.random.shuffle(grid)
    selected_points = grid[:np.floor(len(grid)*fraction).astype(int)]

    groups = []
    for point in selected_points:
        group = [point]
        for _ in range(np.random.randint(1, 5)):  # Randomly create 1 to 4 additional points
            theta = np.random.uniform(0, 360)  # Random direction
            offset_point = point_offset(
                point, np.random.uniform(tolerance*ring_radius[0], tolerance*ring_radius[1]),
                theta)  # Random distance within 1 deg (tolerance)
            group.append(offset_point)
        groups.append(group)
    all_points = np.array([point for group in groups for point in group[0:]])
    return groups, all_points


def check_group_match(expected_groups: list[list[tuple[float, float]]], output_groups):
    """Check if two groups match.

    Parameters
    ----------
    expected_groups : list[list[tuple[float, float]]]
        List of groups that are expected to be grouped correctly.
    output_groups : list[list[tuple[float, float]]]
        List of groups that are output by the tested function.
    
    Returns
    -------
    problematic_groups : list[list[tuple[float, float]]]
        List of groups that the tested function failed to group correctly. If the tested
        function works correctly, an empty list is returned.
    """
    problematic_groups = []
    for expected_group in expected_groups:
        matched = False
        for output_group in output_groups:
            if expected_group[0] in output_group:  # Check if central points match
                if set(expected_group) == set(output_group):  # Check if all points in the group match
                    matched = True
                    break
        if not matched:
            problematic_groups.append(expected_group)
    return problematic_groups


class TestCelestialGrouping_RandomGrid(unittest.TestCase):
    """
    Unit test for a celestial objects grouping method.

    The purpose of this unit test is to ensure that a celestial objects grouping method works correctly. 
    A grid on the celestial sphere is created with a size of 10 deg. Half of the grid points are randomly 
    selected, and for each selected point, several points are created within a 1 deg circle around it using 
    the point_offset() function. These points, with the central point, become a 'group'. The tested function 
    is then called, and its output is checked against the expected groups. If there's a discrepancy, the 
    problematic group is printed.
    """

    def setUp(self):
        seed = np.random.randint(0, 1e5)
        print(f"Seed: {seed}")
        grid = generate_celestial_grid(dec_bounds=70)
        self.tolerance = 1  # deg
        self.expected_groups, self.all_points = create_groups_from_grid(
            grid, self.tolerance, seed=seed, ring_radius=(0.9999, 1.0))

    # @unittest.skip("This test is for the disjoint set method.")
    # def test_group_by_disjoint_set(self):
    #     output_groups = group_by_disjoint_set(self.all_points, self.tolerance)
    #     problematic_groups = check_group_match(self.expected_groups, output_groups)
    #     self.assertEqual(len(problematic_groups), 0, f"Failed groups: {problematic_groups}")

    # @unittest.skip("This test is for the DFS method.")
    # def test_group_by_DFS(self):
    #     output_groups = group_by_DFS(self.all_points, self.tolerance)
    #     problematic_groups = check_group_match(self.expected_groups, output_groups)
    #     self.assertEqual(len(problematic_groups), 0, f"Failed groups: {problematic_groups}")

    def test_group_by_quadtree(self):
        output_groups = fof(self.all_points, self.tolerance).get_coordinates()
        problematic_groups = check_group_match(self.expected_groups, output_groups)
        self.assertEqual(len(problematic_groups), 0, f"Failed groups: {problematic_groups}")


class TestCelestialGrouping_Random(unittest.TestCase):

    # def test_comparing_DFS_quadtree(self):
    #     ra, dec = generate_random_point(10000)
    #     all_points = np.array([ra, dec]).T
    #     tolerance = 2
    #     output_groups_dfs = group_by_DFS(all_points, tolerance)
    #     output_groups_qt = group_by_quadtree(all_points, tolerance).get_coordinates()
    #     problematic_groups = check_group_match(output_groups_dfs, output_groups_qt)
    #     self.assertEqual(len(problematic_groups), 0, f"Failed groups: {problematic_groups}")

    @unittest.skip("This test takes too long to run.")
    def test_comparing_chunk_setting(self):
        ra, dec = generate_random_point(10000, seed=0)
        all_points = np.array([ra, dec]).T
        tolerance = 1.5
        output_groups_base = fof(all_points, tolerance, dec_bound=60, ring_chunk=[6, 6]).get_coordinates()
        for i in range(400):
            print(f"Test {i+1} started!")
            dec = np.random.uniform(50, 80)
            N = np.random.randint(2, 6)
            ring = [np.random.randint(6, 12) for _ in range(N)]
            output_groups_test = fof(all_points, tolerance, dec_bound=dec, ring_chunk=ring).get_coordinates()
            problematic_groups = check_group_match(output_groups_test, output_groups_base)
            self.assertEqual(
                len(problematic_groups),
                0, f"Failed groups: {problematic_groups} with dec_bound={dec}, ring_chunk={ring}")


class TestCelestialGrouping(unittest.TestCase):

    def test_qt_high_density(self):
        grid = generate_celestial_grid(ra_step=1, dec_step=1, dec_bounds=70)
        tolerance = 0.1
        expected_groups, all_points = create_groups_from_grid(grid, tolerance, fraction=0.1, ring_radius=(0.9999, 1.0))
        output_groups = fof(all_points, tolerance).get_coordinates()
        problematic_groups = check_group_match(expected_groups, output_groups)
        self.assertEqual(len(problematic_groups), 0, f"Failed groups: {problematic_groups}")

    def test_qt_grid_boundary(self):
        grid = generate_celestial_grid(ra_step=60, dec_step=5, dec_bounds=60)
        tolerance = 0.2
        expected_groups, all_points = create_groups_from_grid(grid, tolerance, fraction=1)
        output_groups = fof(all_points, tolerance).get_coordinates()
        problematic_groups = check_group_match(expected_groups, output_groups)
        self.assertEqual(len(problematic_groups), 0, f"Failed groups: {problematic_groups}")

    def test_qt_long_chain(self):
        '''
        Test result:
        Found round-off error in the code.
        The problem is fixed by setting a relative tolerance of 1e-8 in the np.isclose() function.
        '''
        dec_range = 15  # Should be an integer multiple of dec_step
        grid = generate_celestial_grid(ra_step=2, dec_step=5, dec_bounds=dec_range, ra_offset=0)
        tolerance = 2
        all_points = np.array(grid)
        output_groups = fof(all_points, tolerance).get_coordinates()
        self.assertEqual(len(output_groups), (dec_range//5)*2+1, f"Number of groups obtained: {len(output_groups)}")

    def test_qt_random_walk(self):
        ra_now = np.random.uniform(0, 360)
        dec_now = np.random.uniform(-90, 90)
        point_now = (ra_now, dec_now)
        all_points = [point_now]
        for _ in range(1000):
            point_now = point_offset(point_now, np.random.uniform(0, 1), np.random.uniform(0, 360))
            all_points.append(point_now)
        all_points = np.array(all_points)
        tolerance = 1
        output_groups = fof(all_points, tolerance).get_coordinates()
        self.assertEqual(len(output_groups), 1, f"Number of groups obtained: {len(output_groups)}")

    def test_qt_random_tree(self):
        ra_now = np.random.uniform(0, 360)
        dec_now = np.random.uniform(-90, 90)
        point_now = (ra_now, dec_now)
        all_points = [point_now]
        for _ in range(1000):
            node = all_points[np.random.randint(0, len(all_points))]
            point_now = point_offset(node, np.random.uniform(0, 1), np.random.uniform(0, 360))
            all_points.append(point_now)
        all_points = np.array(all_points)
        tolerance = 1
        output_groups = fof(all_points, tolerance).get_coordinates()
        self.assertEqual(len(output_groups), 1, f"Number of groups obtained: {len(output_groups)}")


def print_format_group(groups):
    """
    Format a list of celestial groups into the desired format and print them.
    """
    for group in groups:
        central_point_str = f"({group[0][0]:.2f}, {group[0][1]:.2f})"
        surrounding_points_str = ", ".join([f"({point[0]:.2f}, {point[1]:.2f})" for point in group[1:]])
        print(f"[X] {central_point_str}: [{surrounding_points_str}]")


if __name__ == "__main__":
    # unittest.main(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCelestialGrouping_Random)
    unittest.TextTestRunner(verbosity=2).run(suite)
    for i in range(0):
        ut = TestCelestialGrouping_RandomGrid()
        ut.setUp()
        ut.test_group_by_quadtree()
        print(f"Test {i+1} passed!")

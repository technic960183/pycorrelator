import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from collections import defaultdict
import unittest
import numpy as np
import pandas as pd
from pycorrelator import point_offset, generate_random_point
from pycorrelator import xmatch
from test_fof import generate_celestial_grid


def create_catalogs_from_grid(grid, tolerance=1, seed=None, fraction=0.5, ring_radius=(0, 1)):
    """
    Create two catalogs of points from a given grid. The first catalog contains selected central points,
    and the second contains points randomly distributed around those central points within a specified
    distance and angle.

    Parameters:
    - grid (array-like): A list or array of points from which central points are selected.
    - tolerance (float, optional): The maximum radius for creating surrounding points (default is 1).
    - seed (int, optional): An optional random seed for reproducibility (default is None).
    - fraction (float, optional): The fraction of grid points to select as central points (default is 0.5).
    - ring_radius (tuple, optional): A tuple indicating the minimum and maximum fraction of radii used when
      creating surrounding points (default is (0, 1)).

    Operation:
    - If a seed is provided, it is used to initialize the random number generator.
    - The grid is shuffled, and a fraction of the points are selected as central points.
    - For each central point, create a number of surrounding points (between 5 to 9, randomly chosen)
      within the distance defined by the tolerance and ring_radius, and at a random angle.
    - Two catalogs are created: 'catalog1' with central points and 'catalog2' with surrounding points.

    Returns:
    - A tuple containing two elements:
        1. A dictionary with the expected indexes of the surrounding points for each central point.
        2. A tuple with two np.array: 'centrals' (central points) and 'neighbors' (all surrounding
           points for each central point).
    """
    if seed is not None:
        seed = np.random.randint(0, 1e6)
    np.random.seed(seed)
    np.random.shuffle(grid)
    selected_points = grid[:np.floor(len(grid)*fraction).astype(int)]

    centrals = []
    neighbors = []
    expected_idx = {}
    i_cat2 = 0
    for i, point in enumerate(selected_points):
        centrals.append(point)
        idxes = [] # List of indexes of the surrounding points
        for _ in range(np.random.randint(5, 10)):  # Randomly create 1 to 4 additional points
            theta = np.random.uniform(0, 360)  # Random direction
            distance = np.random.uniform(tolerance*ring_radius[0], tolerance*ring_radius[1]) # Random distance
            offset_point = point_offset(point, distance, theta)
            neighbors.append(offset_point)
            idxes.append(i_cat2)
            i_cat2 += 1
        expected_idx[i] = idxes
    centrals_np = np.array([point for point in centrals])
    neighbors_np = np.array([point for point in neighbors])
    return expected_idx, (centrals_np, neighbors_np)


def check_Xmatching(expected_matches: dict, output_matches: defaultdict):
    """
    Compares the expected matching groups with the output matching groups to determine
    if the matching process has been conducted correctly.

    Parameters:
    - expected_matches (dict): A dictionary where keys are central points and values are lists of expected 
                               neighboring points that should match with the central point.
    - output_matches (defaultdict): A defaultdict similar in structure to expected_matches, but contains 
                                    the actual neighboring points matched with each central point by the 
                                    matching algorithm being tested.

    Returns:
    - problematic_matches (list): A list of tuples, where each tuple contains a central point and its expected 
                                  neighboring points that were not matched correctly by the matching algorithm. 
                                  If the algorithm works correctly, this list will be empty.
    """
    problematic_matches = []
    for central, expected_neighbors in expected_matches.items():
        matched = False
        output_neighbors = output_matches[central]
        if len(output_neighbors) != len(expected_neighbors):  # Check if the number of points in the group match
            problematic_matches.append((central, expected_neighbors))
            print(f"Number of points in group {central} does not match!")
            print(f"Expected {len(expected_neighbors)} but got {len(output_neighbors)}")
            continue
        if set(output_neighbors) == set(expected_neighbors):  # Check if all points in the group match
            matched = True
            break
        if not matched:
            problematic_matches.append((central, expected_neighbors))
            print(f"Group {central} does not match!")
    return problematic_matches

def print_format_match(problematic_matches, central_point, surrounding_points):
    for match in problematic_matches:
        p = central_point[match[0]]
        central_point_str = f"({p[0]:.2f}, {p[1]:.2f})"
        pps = surrounding_points[match[1]]
        surrounding_points_str = ", ".join([f"({point[0]:.2f}, {point[1]:.2f})" for point in pps])
        print(f"[X] {central_point_str}: [{surrounding_points_str}]")


class TestCelestialXMatching_RandomGrid(unittest.TestCase):
    """
    A unittest class for verifying the functionality of a celestial object cross-matching algorithm.

    This unit test class is designed to validate the accuracy of a celestial object cross-matching method
    by creating a controlled test environment. It simulates a grid of celestial coordinates, randomly selects
    a subset of these points, and creates groups of points around the selected points within a tolerance radius.
    These groups serve as the 'expected' matches. The cross-matching function under test is then executed, and
    its results are compared to the expected matches to identify any discrepancies.

    Attributes:
        tolerance (float): The tolerance radius within which points are considered to be in the same group,
                           specified in degrees.
        expected_matching (dict): A dictionary containing the expected groups of points, with each group
                                  centered around a selected grid point.
        two_catalogs (tuple of lists): A pair of catalogs generated from the grid; the first catalog contains
                                       the central points, and the second contains points surrounding the
                                       central points.
    """
    def setUp(self):
        # This method will be called before each test, setting up the common resources
        seed = np.random.randint(0, 1e5)
        panda = True
        print(f"Seed: {seed}")
        grid = generate_celestial_grid(dec_bounds=70, ra_step=3, dec_step=3)
        self.tolerance = 0.5  # deg
        self.expected_matching, self.two_catalogs = create_catalogs_from_grid(
            grid, self.tolerance, seed=seed, ring_radius=(0.999, 1.0), fraction=0.8)
        if panda:
            self.two_catalogs = (pd.DataFrame(self.two_catalogs[0], columns=['Ra', 'Dec']),
                                    pd.DataFrame(self.two_catalogs[1], columns=['Ra', 'Dec']))

    def test_match_by_quadtree(self):
        output_matches = xmatch(self.two_catalogs[0], self.two_catalogs[1], self.tolerance).get_result_dict()
        problematic_matches = check_Xmatching(self.expected_matching, output_matches)
        print_format_match(problematic_matches, self.two_catalogs[0], self.two_catalogs[1])
        self.assertEqual(len(problematic_matches), 0, f"Failed groups: {problematic_matches}")

    def test_self_match_by_quadtree(self):
        combine = np.concatenate([self.two_catalogs[1], self.two_catalogs[0]], axis=0)
        output_matches = xmatch(combine, combine, self.tolerance).get_result_dict()
        problematic_matches = []
        err_msg = ""
        for central, expected_neighbors in self.expected_matching.items():
            matched = False
            output_neighbors = output_matches[central + len(self.two_catalogs[1])]
            if len(output_neighbors) != len(expected_neighbors) + 1:  # Check if the number of points in the group match
                problematic_matches.append((central, expected_neighbors))
                print(f"Number of points in group {central} does not match!")
                print(f"Expected {len(expected_neighbors) + 1} but got {len(output_neighbors)}")
                continue
            expected_neighbors.append(central + len(self.two_catalogs[1]))
            if set(output_neighbors) == set(expected_neighbors):  # Check if all points in the group match
                matched = True
                break
            if not matched:
                problematic_matches.append((central, expected_neighbors))
                print(f"Group {central} does not match!")
                err_msg += f"[X] Expected {expected_neighbors} groups but got {output_matches}.\n"
        self.assertEqual(len(problematic_matches), 0, err_msg)


class TestInputFormatXMatch(unittest.TestCase):

    def setUp(self):
        self.tolerance = 3
        self.r1 = [1, 2, 8]
        self.d1 = [3, 4, 6]
        self.r2 = [5, 6, 7]
        self.d2 = [7, 8, 9]

    def test_with_index(self):
        # Test with dataframes that have an index
        df1 = pd.DataFrame({"Ra": self.r1, "Dec": self.d1}, index=[0, 1, 2])
        df2 = pd.DataFrame({"Ra": self.r2, "Dec": self.d2}, index=[0, 1, 2])
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    def test_without_index(self):
        # Test with dataframes without an index
        df1 = pd.DataFrame({"Ra": self.r1, "Dec": self.d1})
        df2 = pd.DataFrame({"Ra": self.r2, "Dec": self.d2})
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    def test_non_successive_index(self):
        # Test with dataframes that have a non-successive index
        df1 = pd.DataFrame({"Ra": self.r1, "Dec": self.d1}, index=[0, 2, 4])
        df2 = pd.DataFrame({"Ra": self.r2, "Dec": self.d2}, index=[1, 3, 5])
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    def test_unsorted_data(self):
        # Test with unsorted data
        df1 = pd.DataFrame({"Ra": self.r1, "Dec": self.d1}, index=[2, 0, 1])
        df2 = pd.DataFrame({"Ra": self.r2, "Dec": self.d2}, index=[8, 7, 4])
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    def test_with_numpy(self):
        # Test with numpy arrays
        df1 = np.array([[1, 3], [2, 4], [8, 6]]) # shape (3, 2)
        df2 = np.array([[5, 7], [6, 8], [7, 9]]) # shape (3, 2)
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    def test_column_names(self):
        # Test with dataframes that have different column names
        df1 = pd.DataFrame({"ra": self.r1, "dec": self.d1}, index=[0, 1, 2])
        df2 = pd.DataFrame({"RA": self.r2, "DEC": self.d2}, index=[0, 1, 2])
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None
        df1 = pd.DataFrame({"ra": self.r1, "dec": self.d1}, index=[0, 1, 2])
        df2 = pd.DataFrame({"RA": self.r2, "Dec": self.d2}, index=[0, 1, 2])
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    @unittest.skip("This feature is considered to be removed.")
    def test_key_with_index(self):
        # Test with dataframes that have a column named 'index' or 'level_0'
        df1 = pd.DataFrame({"Ra": self.r1, "Dec": self.d1, "index": [10, 20, 30]})
        df2 = pd.DataFrame({"Ra": self.r2, "Dec": self.d2, "index": [10, 20, 30], "level_0": [10, 20, 30]})
        try:
            self.result = xmatch(df1, df2, self.tolerance)
            # If no exception is raised, fail the test
            self.fail("ExpectedException not raised")
        except ValueError:
            self.result = {}

    def test_empty_and_one_row(self):
        # Test with empty dataframes
        df1 = pd.DataFrame({"Ra": [], "Dec": []})
        df2 = pd.DataFrame({"Ra": [5], "Dec": [7]})
        self.result = xmatch(df1, df2, self.tolerance)
        self.assertIsNotNone(self.result)  # Assert result is not None

    # Additional tests to consider:
    # - Test with invalid data types
    # - Test with tolerance values (e.g., 0, negative, very large)

    def tearDown(self):
        print(self.result)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCelestialXMatching_RandomGrid)
    unittest.TextTestRunner(verbosity=2).run(suite)
    for i in range(100):
        ut = TestCelestialXMatching_RandomGrid()
        ut.setUp()
        ut.test_match_by_quadtree()
        print(f"Test {i+1} passed!")

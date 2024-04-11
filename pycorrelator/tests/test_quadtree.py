import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import unittest
import numpy as np
from pycorrelator import DataQuadTree

class TestDataQuadTreeInsertion(unittest.TestCase):

    def setUp(self):
        self.tree = DataQuadTree()

    def test_insert_single_data(self):
        self.tree.insert(1, 2, "data1")
        self.assertEqual(self.tree.query(1, 2), ["data1"])

    def test_insert_multiple_data_same_point(self):
        self.tree.insert(1, 2, "data1")
        self.tree.insert(1, 2, "data2")
        self.assertEqual(self.tree.query(1, 2), ["data1", "data2"])

    def test_insert_multiple_data_different_points(self):
        self.tree.insert(1, 2, "data1")
        self.tree.insert(3, 4, "data2")
        self.assertEqual(self.tree.query(1, 2), ["data1"])
        self.assertEqual(self.tree.query(3, 4), ["data2"])

    def test_large_coordinates_insert(self):
        self.tree.insert(int(1e6), int(1e6), "data_large")
        self.assertEqual(self.tree.query(int(1e6), int(1e6)), ["data_large"])
        self.tree.insert(int(-1e6), int(-1e6), "data_small")
        self.assertEqual(self.tree.query(int(-1e6), int(-1e6)), ["data_small"])

    def test_insert_same_point_many_times(self):
        data_list = ["data" + str(i) for i in range(1000)]
        for data in data_list:
            self.tree.insert(0, 0, data)
        self.assertEqual(self.tree.query(0, 0), data_list)

    def test_insert_unique_coordinates_large_number(self):
        coordinates_data = [(i, i, "data" + str(i)) for i in range(1000)]
        random.shuffle(coordinates_data)
        for x, y, data in coordinates_data:
            self.tree.insert(x, y, data)
        for i in range(1000):
            self.assertEqual(self.tree.query(i, i), ["data" + str(i)])


class TestDataQuadTreeQuery(unittest.TestCase):

    def setUp(self):
        self.tree = DataQuadTree()

    def test_query_nonexistent_point(self):
        self.assertEqual(self.tree.query(1, 2), [])

    def test_query_nonexistent_large_coordinates(self):
        self.assertEqual(self.tree.query(int(1e6), int(1e6)), [])
        self.assertEqual(self.tree.query(int(-1e6), int(-1e6)), [])


class TestDataQuadTreeLength(unittest.TestCase):

    def setUp(self):
        self.tree = DataQuadTree()

    def test_length_single_data(self):
        self.tree.insert(1, 2, "data1")
        self.assertEqual(self.tree.length(1, 2), 1)

    def test_length_multiple_data(self):
        self.tree.insert(1, 2, "data1")
        self.tree.insert(1, 2, "data2")
        self.assertEqual(self.tree.length(1, 2), 2)

    def test_length_nonexistent_point(self):
        self.assertEqual(self.tree.length(1, 2), 0)


class TestDataQuadTreeQueryBatch(unittest.TestCase):

    def setUp(self):
        self.tree = DataQuadTree()
        self.tree.insert(1, 1, "Q1")
        self.tree.insert(-1, 1, "Q2")
        self.tree.insert(-1, -1, "Q3")
        self.tree.insert(1, -1, "Q4")

    def test_batch_all_in_same_quadrant(self):
        x_nparray = np.array([1, 1, 1])
        y_nparray = np.array([1, 1, 1])
        result = self.tree.query_batch(x_nparray, y_nparray)
        expected = ["Q1", "Q1", "Q1"]
        self.assertEqual(result, expected)

    def test_batch_in_multiple_quadrants(self):
        x_nparray = np.array([1, -1, -1])
        y_nparray = np.array([1, 1, -1])
        result = self.tree.query_batch(x_nparray, y_nparray)
        expected = ["Q1", "Q2", "Q3"]
        self.assertEqual(result, expected)

    def test_batch_mix_of_empty_and_valid_quadrants(self):
        x_nparray = np.array([2, 1, -1])
        y_nparray = np.array([2, 1, 1])
        result = self.tree.query_batch(x_nparray, y_nparray)
        expected = ["Q1", "Q2"]
        self.assertEqual(result, expected)

    def test_batch_single_point(self):
        x_nparray = np.array([1])
        y_nparray = np.array([1])
        result = self.tree.query_batch(x_nparray, y_nparray)
        self.assertEqual(result, ["Q1"])

    def test_batch_empty_input(self):
        x_nparray = np.array([])
        y_nparray = np.array([])
        result = self.tree.query_batch(x_nparray, y_nparray)
        self.assertEqual(result, [])


class TestDataQuadTreeRandomized(unittest.TestCase):

    def setUp(self):
        self.tree = DataQuadTree()
        self.data_points = []
        # Insert random points into the quadtree
        for _ in range(1000):  # inserting 10000 points
            x = np.random.randint(-100, 100)  # assuming range as -1000 to 1000 for random test
            y = np.random.randint(-100, 100)
            data = f"data_{x}_{y}"
            self.tree.insert(x, y, data)
            self.data_points.append(data)

    def test_random_batch_query(self):
        for _ in range(100):  # running 10 random batch tests
            batch_size = np.random.randint(50, 100)  # random batch size between 1 and 50
            x_nparray = np.random.randint(-100, 100, batch_size)
            y_nparray = np.random.randint(-100, 100, batch_size)
            batch_result = self.tree.query_batch(x_nparray, y_nparray)
            # Validate using individual queries
            expected_result = []
            for x, y in zip(x_nparray, y_nparray):
                expected_result += self.tree.query(x, y)
            self.assertEqual(batch_result, expected_result)

    def test_random_batch_query_central(self):
        for _ in range(100):  # running 10 random batch tests
            batch_size = np.random.randint(50, 100)  # random batch size between 1 and 50
            x_nparray = np.random.randint(-10, 10, batch_size)
            y_nparray = np.random.randint(-10, 10, batch_size)
            batch_result = self.tree.query_batch(x_nparray, y_nparray)
            # Validate using individual queries
            expected_result = []
            for x, y in zip(x_nparray, y_nparray):
                expected_result += self.tree.query(x, y)
            self.assertEqual(batch_result, expected_result)


if __name__ == "__main__":
    unittest.main()

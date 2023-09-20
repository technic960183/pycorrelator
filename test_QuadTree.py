
import unittest
import random
from SphericalMatch.QuadTree import DataQuadTree

class TestDataQuadTreeInsertion(unittest.TestCase):

    def setUp(self):
        self.tree = DataQuadTree()

    def test_insert_single_data(self):
        self.assertFalse(self.tree.insert(1, 2, "data1"))
        self.assertEqual(self.tree.query(1, 2), ["data1"])

    def test_insert_multiple_data_same_point(self):
        self.assertFalse(self.tree.insert(1, 2, "data1"))
        self.tree.insert(1, 2, "data2")
        self.assertEqual(self.tree.query(1, 2), ["data1", "data2"])

    def test_insert_multiple_data_different_points(self):
        self.tree.insert(1, 2, "data1")
        self.tree.insert(3, 4, "data2")
        self.assertEqual(self.tree.query(1, 2), ["data1"])
        self.assertEqual(self.tree.query(3, 4), ["data2"])

    def test_large_coordinates_insert(self):
        self.assertFalse(self.tree.insert(int(1e6), int(1e6), "data_large"))
        self.assertEqual(self.tree.query(int(1e6), int(1e6)), ["data_large"])
        self.assertFalse(self.tree.insert(int(-1e6), int(-1e6), "data_small"))
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


if __name__ == "__main__":
    unittest.main()

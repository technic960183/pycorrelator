from SphericalMatch.QuadTree import DataQuadTree
import random


def test_insert_single_data():
    tree = DataQuadTree()
    assert not tree.insert(1, 2, "data1")
    assert tree.query(1, 2) == ["data1"]
    print("test_insert_single_data passed!")


def test_insert_multiple_data_same_point():
    tree = DataQuadTree()
    assert not tree.insert(1, 2, "data1")
    tree.insert(1, 2, "data2")
    assert tree.query(1, 2) == ["data1", "data2"]
    print("test_insert_multiple_data_same_point passed!")


def test_insert_multiple_data_different_points():
    tree = DataQuadTree()
    tree.insert(1, 2, "data1")
    tree.insert(3, 4, "data2")
    assert tree.query(1, 2) == ["data1"]
    assert tree.query(3, 4) == ["data2"]
    print("test_insert_multiple_data_different_points passed!")


def test_query_nonexistent_point():
    tree = DataQuadTree()
    assert tree.query(1, 2) == []
    print("test_query_nonexistent_point passed!")


def test_length_single_data():
    tree = DataQuadTree()
    tree.insert(1, 2, "data1")
    assert tree.length(1, 2) == 1
    print("test_length_single_data passed!")


def test_length_multiple_data():
    tree = DataQuadTree()
    tree.insert(1, 2, "data1")
    tree.insert(1, 2, "data2")
    assert tree.length(1, 2) == 2
    print("test_length_multiple_data passed!")


def test_length_nonexistent_point():
    tree = DataQuadTree()
    assert tree.length(1, 2) == 0
    print("test_length_nonexistent_point passed!")


def test_large_coordinates_insert():
    tree = DataQuadTree()
    assert not tree.insert(int(1e6), int(1e6), "data_large")
    assert tree.query(int(1e6), int(1e6)) == ["data_large"]
    assert not tree.insert(int(-1e6), int(-1e6), "data_small")
    assert tree.query(int(-1e6), int(-1e6)) == ["data_small"]
    print("test_large_coordinates_insert passed!")


def test_insert_same_point_many_times():
    tree = DataQuadTree()
    data_list = ["data" + str(i) for i in range(1000)]
    for data in data_list:
        tree.insert(0, 0, data)
    assert tree.query(0, 0) == data_list
    print("test_insert_same_point_many_times passed!")


def test_insert_unique_coordinates_large_number():
    tree = DataQuadTree()
    coordinates_data = [(i, i, "data" + str(i)) for i in range(1000)]
    # Shuffle the list to randomize the insertion order
    random.shuffle(coordinates_data)
    for x, y, data in coordinates_data:
        tree.insert(x, y, data)
    for i in range(1000):
        assert tree.query(i, i) == ["data" + str(i)]
    print("test_insert_unique_coordinates_large_number passed!")


def test_query_nonexistent_large_coordinates():
    tree = DataQuadTree()
    assert tree.query(int(1e6), int(1e6)) == []
    assert tree.query(int(-1e6), int(-1e6)) == []
    print("test_query_nonexistent_large_coordinates passed!")


def test_insert_with_non_integer_values():
    tree = DataQuadTree()
    try:
        tree.insert(1.5, 1.5, "data_float")
        assert False, "Expected an error when inserting non-integer coordinates."
    except:
        pass
    print("test_insert_with_non_integer_values passed!")


# Running the tests in the style of test_Toolbox_Spherical.py
if __name__ == "__main__":
    test_insert_single_data()
    test_insert_multiple_data_same_point()
    test_insert_multiple_data_different_points()
    test_query_nonexistent_point()
    test_length_single_data()
    test_length_multiple_data()
    test_length_nonexistent_point()
    test_large_coordinates_insert()
    test_insert_same_point_many_times()
    test_insert_unique_coordinates_large_number()
    test_query_nonexistent_large_coordinates()
    test_insert_with_non_integer_values()
    print("All tests passed.")

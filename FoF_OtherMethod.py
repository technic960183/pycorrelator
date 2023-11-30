from .DisjointSet import DisjointSet
from .Toolbox_Spherical import distances_to_target
import numpy as np
import pandas as pd


def create_adjacency_matrix(objects_array, tolerance):
    """
    Create an adjacency matrix where each entry (i, j) indicates if object i is close enough to object j.

    Parameters:
    - objects_array: numpy array of shape (n, 2) where n is the number of objects. Each row is (RA, DEC) for an object.
    - tolerance: a distance threshold in degrees.

    Returns:
    - adjacency_matrix: A matrix where entry (i, j) is 1 if objects i and j are close based on the tolerance, 0 otherwise.
    """
    num_objects = len(objects_array)
    adjacency_matrix = np.zeros((num_objects, num_objects), dtype=int)

    for i in range(num_objects):
        distances = distances_to_target(objects_array[i], objects_array)
        adjacency_matrix[i] = (distances < tolerance).astype(int)
        adjacency_matrix[i, i] = 0  # set diagonal to 0 as we don't want object to be close to itself

    return adjacency_matrix


def group_close_objects(adjacency_matrix):
    """
    Group objects based on their closeness using the adjacency matrix.

    Parameters:
    - adjacency_matrix: Matrix where entry (i, j) is 1 if objects i and j are close, 0 otherwise.

    Returns:
    - groups: A list of lists, where each inner list contains indices of objects that are close to each other.
    """
    visited = set()
    groups = []

    def dfs(node, current_group):
        """Depth-first search to identify connected components."""
        visited.add(node)
        current_group.append(node)
        for neighbor, is_adjacent in enumerate(adjacency_matrix[node]):
            if is_adjacent and neighbor not in visited:
                dfs(neighbor, current_group)

    for i in range(len(adjacency_matrix)):
        if i not in visited:
            current_group = []
            dfs(i, current_group)
            if current_group:
                groups.append(current_group)

    return groups


def group_by_DFS(objects_array, tolerance):
    if type(objects_array) == pd.DataFrame:
        objects_array = np.array(objects_array[['Ra', 'Dec']].values)
    index = group_close_objects(create_adjacency_matrix(objects_array, tolerance))
    return [[tuple(objects_array[i, :]) for i in g] for g in index]


def group_by_disjoint_set(objects_array, tolerance):
    """
    Group objects based on their closeness using the Disjoint Set (Union-Find) data structure.

    Parameters:
    - objects_array: numpy array of shape (n, 2) where n is the number of objects. Each row is (RA, DEC) for an object.
    - tolerance: a distance threshold in degrees.

    Returns:
    - groups: A list of lists, where each inner list contains indices of objects that are close to each other.
    """
    if type(objects_array) == pd.DataFrame:
        objects_array = np.array(objects_array[['Ra', 'Dec']].values)
    n = objects_array.shape[0]
    ds = DisjointSet(n)

    for i in range(n):
        bool_union = distances_to_target(objects_array[i, :], objects_array) < tolerance
        for j in range(n):
            if bool_union[j]:
                ds.union(i, j)

    groups = ds.get_groups()

    return [[tuple(objects_array[i, :]) for i in g] for g in groups]


def test_grouping_methods(objects_array, tolerance):
    print(f"Testing grouping methods with tolerance of {tolerance} degrees on objects:\n {objects_array}")
    # Group objects using DFS and Disjoint Set methods
    dfs_result = group_close_objects(create_adjacency_matrix(objects_array, tolerance))
    disjoint_set_result = group_by_disjoint_set(objects_array, tolerance)
    print(f"DFS result: {dfs_result}")
    print(f"Disjoint Set result: {disjoint_set_result}")

    # Convert results to sets of sets for easier comparison
    dfs_result = set(frozenset(group) for group in dfs_result)
    disjoint_set_result = set(frozenset(group) for group in disjoint_set_result)

    # Check if both methods produce the same groups
    assert dfs_result == disjoint_set_result, "Groups mismatch"

    print("Both methods produced the same groups. Test passed!")

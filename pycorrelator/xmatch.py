from collections import defaultdict
import numpy as np
from scipy.spatial import KDTree
from .catalog import Catalog
from .chunk import Chunk
from .chunk_generator_grid import GridChunkGenerator
from .euclidean_vs_angular_distance_local import compute_error
from .result_xmatch import XMatchResult
from .utilities_spherical import radec_to_cartesian, cartesian_to_radec
from .utilities_spherical import great_circle_distance, rotate_radec_about_axis
from .utilities_spherical import distances_to_target


def unique_merge_defaultdicts(d1: defaultdict, d2: defaultdict):
    """Joins two dictionaries, merging values for shared keys and preserving others.

    When both dictionaries have the same key, this function makes a new list 
    with every distinct value from either dictionary. If a key is only in one 
    dictionary, it adds that key and its values directly to the result.

    Parameters
    ----------
    d1 : defaultdict
        A dictionary with list-type values.
    d2 : defaultdict
        Another dictionary with list-type values.

    Returns
    -------
    defaultdict
        A dictionary with all keys from both d1 and d2. For shared keys, it has a list
        of unique values. For unshared keys, it has the original list.
    """
    # Convert defaultdicts to arrays
    keys1 = np.array(list(d1.keys()), dtype=np.int64)
    keys2 = np.array(list(d2.keys()), dtype=np.int64)
    # Find intersection and unique keys in both arrays
    intersect_keys = np.intersect1d(keys1, keys2, assume_unique=True)
    unique_keys_d1 = np.setdiff1d(keys1, intersect_keys, assume_unique=True)
    unique_keys_d2 = np.setdiff1d(keys2, intersect_keys, assume_unique=True)
    # Merge the arrays for intersecting keys
    merged_values = [np.union1d(d1[key], d2[key]) for key in intersect_keys]
    # Get the arrays for unique keys
    values_unique_d1 = [d1[key] for key in unique_keys_d1]
    values_unique_d2 = [d2[key] for key in unique_keys_d2]
    # Combine all keys and values
    all_keys = np.concatenate([intersect_keys, unique_keys_d1, unique_keys_d2])
    all_values = merged_values + values_unique_d1 + values_unique_d2
    # Convert back to defaultdict
    result = defaultdict(list, {k: list(v) for k, v in zip(all_keys, all_values)})
    return result

def xmatch(catalog1, catalog2, tolerance, verbose=True) -> XMatchResult:
    """Performs a cross-match between two catalogs.

    This function matches objects from two different catalogs based on their coordinates. Objects from
    `catalog1` and `catalog2` that are within a specified angular distance (tolerance) are considered matches.

    Parameters
    ----------
    catalog1 : array-like
        The first catalog.
    catalog2 : array-like
        The second catalog.
    tolerance : float
        The tolerance for the cross-match in degrees.
    verbose : bool, optional
        Whether to print the progress.

    Returns
    -------
    XMatchResult
        A XMatchResult object that contains the cross-match result.
    """
    # [ENH]: Add an option for sorting the output
    _catalog1 = Catalog(catalog1)
    _catalog2 = Catalog(catalog2)
    cg1 = GridChunkGenerator(margin=2*tolerance)
    cg2 = GridChunkGenerator(margin=2*tolerance)
    cg1.set_symmetric_ring_chunk(60, [6, 6])
    cg2.set_symmetric_ring_chunk(60, [6, 6])
    cg1.distribute(_catalog1)
    cg2.distribute(_catalog2)
    if len(cg1.chunks) != len(cg2.chunks):
        raise BrokenPipeError("The two catalogs have different number of chunks! Please contact the developer.")
    merged_dict = defaultdict(list) # [FIXME] Change to dict or sorted dict, or don't assume the order of the keys.
    for i in range(len(cg1.chunks)):
        if verbose:
            print(f"Started Chunk {i}")
        dd = xmatch_chunk((cg1.chunks[i], cg2.chunks[i], tolerance))
        if i == 0:
            merged_dict = dd
        else:
            merged_dict = unique_merge_defaultdicts(merged_dict, dd)
    return XMatchResult(_catalog1, _catalog2, tolerance, merged_dict)

def rotate_to_center(object_coor, chunk_ra, chunk_dec):
    # Rotate the center of the chunk to (180, 0) of the celestial sphere
    center_car = radec_to_cartesian(chunk_ra, chunk_dec)
    normal_car = np.cross(center_car, np.array([-1., 0., 0.]))
    normal_car /= np.linalg.norm(normal_car)
    normal_ra, normal_dec = cartesian_to_radec(normal_car)
    angle = great_circle_distance(chunk_ra, chunk_dec, 180, 0)
    rot_ra, rot_dec = rotate_radec_about_axis(object_coor[:,0], object_coor[:,1], normal_ra, normal_dec, angle)
    return rot_ra, rot_dec

def xmatch_chunk(args: tuple[Chunk, Chunk, float]):
    chunk1, chunk2, tolerance = args
    objects1, objects2 = chunk1.get_data(), chunk2.get_data()
    index1, index2 = chunk1.get_index(), chunk2.get_index()
    if chunk1.get_center() != chunk2.get_center():
        raise ValueError("The two chunks have different centers!")
    ra, dec = chunk1.get_center()
    rot_coor1 = np.array(rotate_to_center(objects1, ra, dec)).T
    rot_coor2 = np.array(rotate_to_center(objects2, ra, dec)).T
    if chunk1.farest_distance() != chunk2.farest_distance():
        raise ValueError("The two chunks have different farest distances!")
    SAFTY_FACTOR = 1.01
    A2E_factor = (1 + compute_error(chunk1.farest_distance(), tolerance)) * SAFTY_FACTOR
    idx1, idxes2 = spherical_xmatching(index1, rot_coor1, index2, rot_coor2, tolerance, A2E_factor)
    dd = defaultdict(list)
    for key, value in zip(idx1, idxes2):
        dd[key] = value
    return dd

def spherical_xmatching(idx1: np.array, coor1: np.array, idx2: np.array, coor2: np.array, tolerance, A2E_factor):
    qt1 = KDTree(coor1)
    qt2 = KDTree(coor2)
    list_of_indexes = qt1.query_ball_tree(qt2, tolerance * A2E_factor) # list of elements in idx2
    keys, vals = [], []
    for i, indexes in enumerate(list_of_indexes):
        distance = distances_to_target(coor1[i, :], coor2[indexes, :])
        is_close = (distance < tolerance) | np.isclose(distance, tolerance, rtol=1e-8)
        keys.append(idx1[i])
        vals.append(idx2[indexes][is_close])
    return keys, vals

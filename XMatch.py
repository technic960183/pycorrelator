import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from .ChunkGenerator_Grid import GridChunkGenerator
from .XMatchResult import XMatchResult
from .Toolbox_Spherical import radec_to_cartesian, cartesian_to_radec
from .Toolbox_Spherical import great_circle_distance, rotate_radec_about_axis
from .Toolbox_Spherical import distances_to_target
from .EuclideanVsAngularDistanceAnalysis_Local import compute_error


def unique_merge_defaultdicts(d1: defaultdict, d2: defaultdict):
    """
    Joins two dictionaries, merging values for shared keys and preserving others.

    When both dictionaries have the same key, this function makes a new list 
    with every distinct value from either dictionary. If a key is only in one 
    dictionary, it adds that key and its values directly to the result.

    Parameters:
    - d1 (defaultdict): A dictionary with list-type values.
    - d2 (defaultdict): Another dictionary with list-type values.

    Returns:
    - defaultdict: A dictionary with all keys from both d1 and d2. For shared keys,
      it has a list of unique values. For unshared keys, it has the original list.
    """
    # Convert defaultdicts to arrays
    keys1 = np.array(list(d1.keys()))
    keys2 = np.array(list(d2.keys()))
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

def Verify_Input(data: pd.DataFrame | np.ndarray, retain_index: bool):
    if type(data) == np.ndarray:
        if data.shape[1] != 2:
            raise ValueError("The input array must have two columns!")
        data = pd.DataFrame(data, columns=['Ra', 'Dec'])
    elif type(data) != pd.DataFrame:
        raise ValueError("The input data must be a numpy array or a pandas dataframe!")
    ras = ['ra', 'Ra', 'RA']
    decs = ['dec', 'Dec', 'DEC']
    # count the number of columns that contain 'ra' or 'dec'
    hit_ra = np.array([1 if col in ras else 0 for col in data.columns])
    hit_dec = np.array([1 if col in decs else 0 for col in data.columns])
    if sum(hit_ra) != 1 or sum(hit_dec) != 1:
        raise ValueError("The input dataframe must have two columns named 'ra' and 'dec'!")
    # rename the columns to 'Ra' and 'Dec'
    if data.columns[hit_ra == 1][0] != 'Ra':
        data.rename(columns={data.columns[hit_ra == 1][0]: 'Ra'}, inplace=True)
    if data.columns[hit_dec == 1][0] != 'Dec':
        data.rename(columns={data.columns[hit_dec == 1][0]: 'Dec'}, inplace=True)
    # check if columns 'index' exist
    # [BUG] possible bug: MultiIndex index
    if 'index' in data.columns:
        raise ValueError("The input dataframe must not have a column named 'index'!")
    data.reset_index(inplace=True, drop=not retain_index)
    if retain_index:
        data.rename(columns={'index': 'original_index'}, inplace=True)
    data.reset_index(inplace=True)
    return data

def XMatch(df1: pd.DataFrame, df2: pd.DataFrame, tolerance, retain_index=False, inplace=False):
    """
    Purpose: This function performs a cross-match between two catalogs.
    Parameters:
        - objects_df1 (DataFrame): The first catalog.
        - objects_df2 (DataFrame): The second catalog.
        - tolerance (float): The tolerance for the cross-match in degrees.
        - retain_index (bool): Whether to retain the index in the input.
        - inplace (bool): Whether to perform the cross-match inplace. If True, the
            function will modify the input catalogs.
    Returns:
        - XMatchResult: A XMatchResult object that contains the cross-match result.
    """
    if not inplace:
        df1 = df1.copy()
        df2 = df2.copy()
    df1 = Verify_Input(df1, retain_index)
    df2 = Verify_Input(df2, retain_index)
    cg1 = GridChunkGenerator(margin=2*tolerance)
    cg2 = GridChunkGenerator(margin=2*tolerance)
    cg1.set_symmetric_ring_chunk(60, [6, 6])
    cg2.set_symmetric_ring_chunk(60, [6, 6])
    cg1.distribute(df1)
    cg2.distribute(df2)
    if len(cg1.chunks) != len(cg2.chunks):
        raise ValueError("The two catalogs have different number of chunks!")
    merged_dict = defaultdict(list)
    for i in range(len(cg1.chunks)):
        print(f"Started Chunk {i}")
        dd = XMatch_chunk((cg1.chunks[i], cg2.chunks[i], tolerance))
        if i == 0:
            merged_dict = dd
        else:
            merged_dict = unique_merge_defaultdicts(merged_dict, dd)
    merged_dict = {k: v for k, v in merged_dict.items() if len(v) != 0} # Remove keys with empty values
    return XMatchResult(df1, df2, tolerance, merged_dict)

def rotate_to_center(object_df, ra, dec):
    # Rotate the center of the chunk to (180, 0) of the celestial sphere
    center_car = radec_to_cartesian(ra, dec)
    normal_car = np.cross(center_car, np.array([-1., 0., 0.]))
    normal_car /= np.linalg.norm(normal_car)
    normal_ra, normal_dec = cartesian_to_radec(normal_car)
    angle = great_circle_distance(ra, dec, 180, 0)
    rot_ra, rot_dec = rotate_radec_about_axis(object_df['Ra'], object_df['Dec'], normal_ra, normal_dec, angle)
    return rot_ra, rot_dec

def XMatch_chunk(args):
    chunk1, chunk2, tolerance = args
    objects1, objects2 = chunk1.get_data(), chunk2.get_data()
    if chunk1.get_center() != chunk2.get_center():
        raise ValueError("The two chunks have different centers!")
    ra, dec = chunk1.get_center()
    rot_coor1 = np.array(rotate_to_center(objects1, ra, dec)).T
    rot_coor2 = np.array(rotate_to_center(objects2, ra, dec)).T
    index1 = objects1['index'].values
    index2 = objects2['index'].values
    if chunk1.farest_distance() != chunk2.farest_distance():
        raise ValueError("The two chunks have different farest distances!")
    SAFTY_FACTOR = 1.01
    A2E_factor = (1 + compute_error(chunk1.farest_distance(), tolerance)) * SAFTY_FACTOR
    idx1, idxes2 = spherical_Xmatching(index1, rot_coor1, index2, rot_coor2, tolerance, A2E_factor)
    dd = defaultdict(list)
    for key, value in zip(idx1, idxes2):
        dd[key] = value
    return dd

def spherical_Xmatching(idx1: np.array, coor1: np.array, idx2: np.array, coor2: np.array, tolerance, A2E_factor):
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
import numpy as np
import pandas as pd
from .chunk_generator_grid import ChunkGeneratorByGrid
from .convex_hull import inner_grid_points
from .disjoint_set import DisjointSet
from .quadtree import DataQuadTree
from .result_fof import FoFResult
from .utilities_spherical import radec_to_cartesian, cartesian_to_radec
from .utilities_spherical import great_circle_distance, rotate_radec_about_axis
from .utilities_spherical import point_offset


def map_to_discrete_grid(n_array, b):
    return np.round(n_array / b).astype(int)


def group_by_quadtree(objects_df: pd.DataFrame, tolerance):
    if type(objects_df) == np.ndarray:
        objects_df = pd.DataFrame(objects_df, columns=['Ra', 'Dec'])
    objects_df.reset_index(inplace=True)
    CG = ChunkGeneratorByGrid(margin=tolerance)
    CG.distribute(objects_df)
    ds = DisjointSet(len(objects_df))
    print(f"Using single process to group {len(CG.chunks)} chunks.")
    for chunk in CG.chunks:
        groups_index = group_by_quadtree_chunk((chunk, tolerance))
        for i, j in groups_index:
            ds.union(i, j)
    groups = ds.get_groups()
    # objects_coordinates = objects_df[['Ra', 'Dec']].values
    # return [[tuple(objects_coordinates[i, :]) for i in g] for g in groups]
    return FoFResult(objects_df, tolerance, groups)


def group_by_quadtree_chunk(args):
    chunk, tolerance = args
    objects = chunk.get_data()
    # Rotate the center of the chunk to (180, 0) of the celestial sphere
    ra, dec = chunk.get_center()
    center_car = radec_to_cartesian(ra, dec)
    normal_car = np.cross(center_car, np.array([-1., 0., 0.]))
    normal_car /= np.linalg.norm(normal_car)
    normal_ra, normal_dec = cartesian_to_radec(normal_car)
    angle = great_circle_distance(ra, dec, 180, 0)
    rot_ra, rot_dec = rotate_radec_about_axis(objects['Ra'], objects['Dec'], normal_ra, normal_dec, angle)
    objects['rot_Ra'] = rot_ra
    objects['rot_Dec'] = rot_dec
    objects['grid_Ra'] = map_to_discrete_grid(rot_ra, tolerance)
    objects['grid_Dec'] = map_to_discrete_grid(rot_dec, tolerance)
    grid_np = objects[['grid_Ra', 'grid_Dec']].values
    corrdinates_np = objects[['rot_Ra', 'rot_Dec']].values
    index_np = objects['index'].values
    groups_index = spherical_quadtree_grouping(index_np, grid_np, corrdinates_np, tolerance)
    return groups_index


def spherical_quadtree_grouping(original_indexes: np.array, grid: np.array, coordinate: np.array, tolerance):
    # Calculate the side length of the octagon
    s = 2 * 1 * np.sin(np.radians(45) / 2)
    # Calculate the height of the right triangle, which is also the inradius of the octagon
    h = np.sqrt(1 - (s/2)**2)
    # 1.5 is a magic number. 1.3 is the known lower bound (200 trails) for chunk with 41.4 deg radius.
    factor = 1 / h * 1.5
    directions = np.arange(0, 360, 45)
    rtn = []
    Dqt = DataQuadTree()
    for i in range(grid.shape[0]):
        Dqt.insert(grid[i, 0],   # grid_Ra
                   grid[i, 1],   # grid_Dec
                   i)   # index
        # To check the neighbors
        # 1. Get the circle on the celestial sphere
        probe_Ra, probe_Dec = point_offset((coordinate[i, 0], coordinate[i, 1]), tolerance * factor, directions)
        # 2. project the circle to the discretized grid (the projected curve should be CONVEX)
        probe_grid_Ra = map_to_discrete_grid(probe_Ra, tolerance)
        probe_grid_Dec = map_to_discrete_grid(probe_Dec, tolerance)
        # 3. Pick up the inner missing grid points by the upper and lower bounds of the circle (May be optimized)
        inner_grid = inner_grid_points(np.array([probe_grid_Ra, probe_grid_Dec]).T)
        # 4. Check the objects of the grid points as neighbors by calculating the great circle distance
        indexes = []
        for r, d in inner_grid:
            indexes += Dqt.query(r, d)   # Expensive operation, should support batch query.
        indexes = np.unique(indexes)
        indexes = indexes[indexes != i]
        for j in indexes:
            distance = great_circle_distance(coordinate[j, 0], coordinate[j, 1], coordinate[i, 0], coordinate[i, 1])
            if distance < tolerance or np.isclose(distance, tolerance, rtol=1e-8):
                rtn.append((original_indexes[i], original_indexes[j]))
    return rtn

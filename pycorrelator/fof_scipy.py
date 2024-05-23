import numpy as np
from scipy.spatial import KDTree
from .catalog import Catalog
from .chunk import Chunk
from .chunk_generator_grid import GridChunkGenerator
from .disjoint_set import DisjointSet
from .euclidean_vs_angular_distance_local import compute_error
from .result_fof import FoFResult
from .utilities_spherical import radec_to_cartesian, cartesian_to_radec
from .utilities_spherical import great_circle_distance, rotate_radec_about_axis


def group_by_quadtree(catalog, tolerance, dec_bound=60, ring_chunk=[6, 6]):
    _catalog = Catalog(catalog)
    cg = GridChunkGenerator(margin=2*tolerance)
    cg.set_symmetric_ring_chunk(dec_bound, ring_chunk)
    cg.distribute(_catalog)
    print(f"[Scipy Version] Using single process to group {len(cg.chunks)} chunks.")
    ds = DisjointSet(len(_catalog))
    for chunk in cg.chunks:
        groups_index = group_by_quadtree_chunk((chunk, tolerance))
        for i, j in groups_index:
            ds.union(i, j)
    groups = ds.get_groups()
    return FoFResult(_catalog, tolerance, groups)


def group_by_quadtree_chunk(args: tuple[Chunk, float]):
    chunk, tolerance = args
    objects = chunk.get_data()
    # Rotate the center of the chunk to (180, 0) of the celestial sphere
    ra, dec = chunk.get_center()
    center_car = radec_to_cartesian(ra, dec)
    normal_car = np.cross(center_car, np.array([-1., 0., 0.]))
    normal_car /= np.linalg.norm(normal_car)
    normal_ra, normal_dec = cartesian_to_radec(normal_car)
    angle = great_circle_distance(ra, dec, 180, 0)
    rot_ra, rot_dec = rotate_radec_about_axis(objects[:, 0], objects[:, 1], normal_ra, normal_dec, angle)
    corrdinates_np = np.vstack((rot_ra, rot_dec)).T
    index_np = chunk.get_index()
    SAFTY_FACTOR = 1.05
    A2E_factor = (1 + compute_error(chunk.farest_distance(), tolerance)) * SAFTY_FACTOR
    groups_index = spherical_quadtree_grouping(index_np, corrdinates_np, tolerance, A2E_factor)
    return groups_index


def spherical_quadtree_grouping(original_indexes: np.array, coordinate: np.array, tolerance, A2E_factor):
    qt = KDTree(coordinate)
    indexes = qt.query_pairs(tolerance * A2E_factor)
    rtn = []
    for i in indexes:
        distance = great_circle_distance(coordinate[i[0], 0], coordinate[i[0], 1],
                                         coordinate[i[1], 0], coordinate[i[1], 1])
        if distance < tolerance or np.isclose(distance, tolerance, rtol=1e-8):
            rtn.append((original_indexes[i[0]], original_indexes[i[1]]))
    return rtn

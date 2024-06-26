import multiprocessing
import numpy as np
import pandas as pd
from .chunk_generator_grid import ChunkGeneratorByGrid, ChunkGeneratorBySuperDenseGrid
from .disjoint_set import DisjointSet
from .fof_scipy import group_by_quadtree_chunk
from .result_fof import FoFResult


def group_by_quadtree(objects_df: pd.DataFrame, tolerance):
    if type(objects_df) == np.ndarray:
        objects_df = pd.DataFrame(objects_df, columns=['Ra', 'Dec'])
    objects_df.reset_index(inplace=True)
    # CG = ChunkGeneratorByGrid(margin=tolerance)
    CG = ChunkGeneratorBySuperDenseGrid(margin=tolerance)
    CG.distribute(objects_df)
    ds = DisjointSet(len(objects_df))
    print(f"Using {multiprocessing.cpu_count()} processes to group {len(CG.chunks)} chunks.")
    with multiprocessing.Pool() as pool:
        # Pack chunk and tolerance into a tuple for Pool.map
        all_groups_index = pool.map(group_by_quadtree_chunk, [(chunk, tolerance) for chunk in CG.chunks])

    for groups_index in all_groups_index:
        for i, j in groups_index:
            ds.union(i, j)

    groups = ds.get_groups()
    # objects_coordinates = objects_df[['Ra', 'Dec']].values
    # return [[tuple(objects_coordinates[i, :]) for i in g] for g in groups]
    return FoFResult(objects_df, tolerance, groups)

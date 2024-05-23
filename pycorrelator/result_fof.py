import pandas as pd
import numpy as np
from .catalog import Catalog

class FoFResult:
    def __init__(self, catalog: Catalog, tolerance, result_list: list):
        self.catalog = catalog
        self.tolerance = tolerance
        self.result_list = result_list

    def get_coordinates(self):
        """
        Returns a list of lists of tuples of coordinates of objects in each group.
        """
        objects_coordinates = self.catalog.get_coordiantes()
        return [[tuple(objects_coordinates[i, :]) for i in g] for g in self.result_list]
    
    def get_group_coordinates(self):
        """
        Returns a list of tuples of coordinates of the groups.
        """
        objects_coordinates = self.catalog.get_coordiantes()
        return [np.average(objects_coordinates[g, :], axis=0) for g in self.result_list]
    
    def get_group_dataframe(self, min_group_size=1):
        raise BrokenPipeError("The method need to be fixed.")
        # [TODO] Modify the method to adapt to the new catalog structure
        new_index_tuples = []
        original_indices = []

        for group_index, group_indices in enumerate(self.result_list):
            # Skip groups smaller than the specified min_group_size
            if len(group_indices) < min_group_size:
                continue

            for object_index in group_indices:
                new_index_tuples.append((group_index, object_index))
                original_indices.append(object_index)

        new_index = pd.MultiIndex.from_tuples(new_index_tuples, names=['Group', 'Object'])
        grouped_df = self.catalog.loc[original_indices].copy()
        grouped_df.index = new_index

        return grouped_df

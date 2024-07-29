import pandas as pd
import numpy as np
from .catalog import Catalog

class FoFResult:
    
    def __init__(self, catalog: Catalog, tolerance: float, result_list: list):
        self.catalog = catalog
        self.tolerance = tolerance
        self.result_list = result_list

    def get_coordinates(self) -> list[list[tuple]]:
        """Returns the coordinates of objects grouped as lists of tuples.

        Returns
        -------
        list[list[tuple]]
            A list of lists of tuples of coordinates of objects in each group.
        """
        objects_coordinates = self.catalog.get_coordiantes()
        return [[tuple(objects_coordinates[i, :]) for i in g] for g in self.result_list]
    
    def get_group_coordinates(self) -> list[tuple]:
        """Returns the center coordinates of the groups.

        Returns
        -------
        list[tuple]
            A list of tuples of coordinates of the center of each group.
        """
        objects_coordinates = self.catalog.get_coordiantes()
        # [FIXME] This return a list of NDArrays, not a list of tuples.
        return [np.average(objects_coordinates[g, :], axis=0) for g in self.result_list]
    
    def get_group_sizes(self) -> list[int]:
        """Returns the object counts in each group.

        Returns
        -------
        list[int]
            A list of integers representing the number of objects in each group.
        """
        return [len(g) for g in self.result_list]
    
    def get_group_dataframe(self, min_group_size=1, coord_columns=['Ra', 'Dec'],
                            retain_all_columns=True, retain_columns=None) -> pd.DataFrame:
        """Get the grouped data as a two-level indexed pandas DataFrame.

        Parameters
        ----------
        min_group_size : int, optional
            The minimum group size to include in the DataFrame. Default is 1.
        coord_columns : list[str], optional
            The names of the columns for the coordinates. Default is ['Ra', 'Dec'].
        retain_all_columns : bool, optional
            Whether to retain all the columns in the input (dataframe). Default is True.
        retain_columns : list[str], optional
            The names of the columns to retain in the output dataframe. Will override retain_all_columns if not empty.
            Default is None.

        Returns
        -------
        pandas.DataFrame
            A two-level indexed pandas DataFrame containing the grouped data.
        """
        new_index_tuples = []
        original_indices = []
        for group_index, group_indices in enumerate(self.result_list):
            if len(group_indices) < min_group_size: # Skip groups with the size less than min_group_size
                continue
            for object_index in group_indices:
                new_index_tuples.append((group_index, object_index))
                original_indices.append(object_index)

        data_df = pd.DataFrame(self.catalog.get_coordiantes(), columns=coord_columns, index=self.catalog.get_indexes())
        append_df = self.catalog.get_appending_data(retain_all_columns, retain_columns)
        if len(append_df.columns) > 0:
            data_df = pd.concat([data_df, append_df], axis=1)
        grouped_df = data_df.iloc[original_indices].copy()
        grouped_df.index = pd.MultiIndex.from_tuples(new_index_tuples, names=['Group', 'Object'])
        return grouped_df

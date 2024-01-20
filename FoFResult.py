import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
from ..objects.FileIO import WriteTractor

class FoFResult:
    def __init__(self, df: pd.DataFrame, tolerance, result_list: list):
        self.df = df
        self.tolerance = tolerance
        self.result_list = result_list

    def get_coordinates(self):
        """
        Returns a list of lists of tuples of coordinates of objects in each group.
        """
        objects_coordinates = self.df[['Ra', 'Dec']].values
        return [[tuple(objects_coordinates[i, :]) for i in g] for g in self.result_list]
    
    def get_group_coordinates(self):
        """
        Returns a list of tuples of coordinates of the groups.
        """
        objects_coordinates = self.df[['Ra', 'Dec']].values
        return [np.average(objects_coordinates[g, :], axis=0) for g in self.result_list]
    
    def get_group_dataframe(self, min_group_size=1):
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
        grouped_df = self.df.loc[original_indices].copy()
        grouped_df.index = new_index

        return grouped_df

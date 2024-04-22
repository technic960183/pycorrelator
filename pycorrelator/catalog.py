import numpy as np
import pandas as pd


class Catalog:

    def __init__(self, data):
        self.datatype = type(data)
        self.input_data = data
        if self.datatype == np.ndarray:
            self.__type_np_array()
        elif self.datatype == pd.DataFrame:
            self.__type_pd_dataframe()

    def get_dataframe(self):
        return self.data
        
    def __type_np_array(self):
        if self.input_data.shape[1] != 2:
            raise ValueError("The input array must have two columns!")
        self.data = pd.DataFrame(self.input_data, columns=['Ra', 'Dec'])
        self.data.reset_index(inplace=True)
            
    def __type_pd_dataframe(self):
        ras = ['ra', 'Ra', 'RA']
        decs = ['dec', 'Dec', 'DEC']
        # Find the location of columns that named 'Ra's or 'Dec's
        hit_ra = np.array([1 if col in ras else 0 for col in self.input_data.columns])
        hit_dec = np.array([1 if col in decs else 0 for col in self.input_data.columns])
        if sum(hit_ra) != 1 or sum(hit_dec) != 1:
            raise ValueError("The input dataframe must have two columns named 'Ra' and 'Dec'!")
        self.data = self.input_data.copy()
        # rename the columns to 'Ra' and 'Dec'
        if self.data.columns[hit_ra == 1][0] != 'Ra':
            self.data.rename(columns={self.data.columns[hit_ra == 1][0]: 'Ra'}, inplace=True)
        if self.data.columns[hit_dec == 1][0] != 'Dec':
            self.data.rename(columns={self.data.columns[hit_dec == 1][0]: 'Dec'}, inplace=True)
        if 'index' in self.data.columns:
            raise ValueError("The input dataframe must not have a column named 'index'!")
        self.data.reset_index(inplace=True, drop=False)
        self.data.rename(columns={'index': 'original_index'}, inplace=True)
        self.data.reset_index(inplace=True)
        
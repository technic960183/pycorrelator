import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Catalog:
    '''
    Purpose: This class is used to store and manipulate the catalog data for xmatch and fof.
    Parameters:
        - data (array-like): The input data can be either a numpy array or a pandas dataframe.
          * np.array: The array must have a shape of (N, 2), representing N points with
            two values: [ra (azimuth, longitude), dec (alltitude, latitude)].
          * pd.DataFrame: The dataframe must have two columns named 'Ra' and 'Dec' (plus all the
            possible combinations with 'ra', 'dec'; 'RA', 'DEC').
    '''

    def __init__(self, data):
        self.datatype = type(data)
        self.input_data = data
        self.ra = None # ra, longitude, azimuth
        self.dec = None # dec, latitude, alltitude
        if self.datatype == np.ndarray:
            self.__type_np_array()
        elif self.datatype == pd.DataFrame:
            self.__type_pd_dataframe()
        elif self.datatype == tuple:
            raise NotImplementedError() # [TODO] Support tuple input for Catalog
        elif self.datatype == list:
            raise NotImplementedError() # [TODO] Support list input for Catalog
        elif self.datatype == dict:
            raise NotImplementedError() # [TODO] Support dict input for Catalog
        else:
            raise TypeError("The input data must be either a numpy array or a pandas dataframe!")
        self._check_validity_range()
        
    def _check_validity_range(self):
        '''
        Purpose: Check the validity of the input data. Warning if the data is out of range.
        '''
        if np.any(self.ra < 0) or np.any(self.ra > 360):
            print("Warning: Ra values are out of range [0, 360]!")
        if np.any(self.dec < -90) or np.any(self.dec > 90):
            print("Warning: Dec values are out of range [-90, 90]!")
        if np.isnan(self.ra).any() or np.isnan(self.dec).any():
            raise ValueError("Input data contains NaN values!")
        if np.isinf(self.ra).any() or np.isinf(self.dec).any():
            raise ValueError("Input data contains Inf values!")
        if len(self.ra) != len(self.dec):
            raise ValueError("The length of Ra and Dec must be the same!")


    def get_coordiantes(self) -> NDArray[np.float64]:
        '''
        Purpose: Get the coordinate of the points in the catalog for xmatch and fof.
        Returns:
            - np.ndarray: The array of shape (N, 2) with [Ra, Dec].
        '''
        return np.vstack([self.ra, self.dec], dtype=np.float64).T
    
    def get_indexes(self) -> NDArray[np.int64]:
        '''
        Purpose: Get the indexes of the points in the catalog for xmatch and fof.
        Returns:
            - np.ndarray: The array of indexes of shape (N,).
        '''
        return np.arange(len(self.ra), dtype=np.int64)
        
    def __type_np_array(self):
        if self.input_data.ndim != 2:
            raise ValueError("The input array must be two-dimensional!")
        if self.input_data.shape[1] != 2:
            raise ValueError("The input array must have two columns!")
        self.ra = self.input_data[:, 0]
        self.dec = self.input_data[:, 1]
            
    def __type_pd_dataframe(self):
        RAS = ['ra', 'Ra', 'RA']
        DECS = ['dec', 'Dec', 'DEC']
        # Find the location of columns that named 'Ra's or 'Dec's
        hit_ra = np.array([1 if col in RAS else 0 for col in self.input_data.columns])
        hit_dec = np.array([1 if col in DECS else 0 for col in self.input_data.columns])
        if sum(hit_ra) != 1 or sum(hit_dec) != 1:
            raise ValueError("The input dataframe must have two columns named 'Ra' and 'Dec'!")
        self.ra = self.input_data[self.input_data.columns[hit_ra == 1][0]].values
        self.dec = self.input_data[self.input_data.columns[hit_dec == 1][0]].values

    def __len__(self):
        return len(self.ra)

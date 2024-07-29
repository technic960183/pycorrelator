from typing import Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Catalog:
    '''This class is used to store and manipulate the catalog data for xmatch and fof.

    Parameters
    ----------    
    data : array-like
        The input data can be either a numpy array or a pandas dataframe.

        * np.array: The array must have a shape of (N, 2), representing N points with
          two values: [ra (azimuth, longitude), dec (alltitude, latitude)].
        * pd.DataFrame: The dataframe must have two columns named 'Ra' and 'Dec' (or all the
          possible combinations with 'ra', 'dec'; 'RA', 'DEC').
    
    '''

    def __init__(self, data):
        self.datatype = type(data)
        self.input_data = data
        self.ra = None # ra, longitude, azimuth
        self.dec = None # dec, latitude, alltitude
        self.ra_column: Optional[str] = None
        self.dec_column: Optional[str] = None
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
        '''Check the validity of the input data. Warning if the data is out of range.
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
        '''Get the coordinate of the points in the catalog for xmatch and fof.

        Returns
        -------
        numpy.ndarray
            The array of shape (N, 2) with [Ra, Dec].
        '''
        return np.vstack([self.ra, self.dec], dtype=np.float64).T
    
    def get_indexes(self) -> NDArray[np.int64]:
        '''Get the indexes of the points in the catalog for xmatch and fof.

        Returns
        -------
        numpy.ndarray
            The array of indexes of shape (N,).
        '''
        return np.arange(len(self.ra), dtype=np.int64)
    
    def get_appending_data(self, retain_all_columns=True, retain_columns=None,
                           invalid_key_error=True) -> pd.DataFrame:
        '''Get the appending data of the points in the catalog for xmatch and fof.

        Parameters
        ----------
        retain_all_columns : bool, optional
            Whether to retain all the columns in the input dataframe. Default is True.
        retain_columns : list, optional
            The list of columns to retain in the input dataframe. Overrides retain_all_columns if not empty.
        invalid_key_error : bool, optional
            Whether to raise an error when the columns are not in the input dataframe. Default is True.
        
        Returns
        -------
        pandas.DataFrame
            The dataframe of the appending data.
        '''
        if self.datatype != pd.DataFrame:
            return pd.DataFrame(index=self.get_indexes())
        columns = []
        if retain_all_columns:
            columns = list(self.input_data.columns)
        if retain_columns is not None:
            if isinstance(retain_columns, list) and len(retain_columns) > 0:
                if all(isinstance(col, str) for col in retain_columns):
                    columns: list[str] = retain_columns
                else:
                    raise TypeError("The elements in retain_columns must be string of column names!")
            elif isinstance(retain_columns, str):
                raise TypeError(f"Cannot accept a string for retain_columns. Please provide it as a list: ['{retain_columns}']")
            else:
                raise TypeError(f"Invalid type for retain_columns: {type(retain_columns)}")
        # Check if the columns are in the input DataFrame
        non_existent_columns = [col for col in columns if col not in self.input_data.columns]  
        if non_existent_columns and invalid_key_error:
            raise KeyError(f"Columns {non_existent_columns} are not in the input DataFrame")
        if not invalid_key_error: # Need to remove the non-existent columns only when invalid_key_error is False
            columns = [col for col in columns if col in self.input_data.columns]
        # Drop the ra and dec columns
        if self.ra_column is not None and self.ra_column in columns:
            columns.remove(self.ra_column)
        if self.dec_column is not None and self.dec_column in columns:
            columns.remove(self.dec_column)
        return pd.DataFrame(self.input_data[columns].values, index=self.get_indexes(), columns=columns)
        
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
        self.ra_column = self.input_data.columns[hit_ra == 1][0]
        self.dec_column = self.input_data.columns[hit_dec == 1][0]
        self.ra = self.input_data[self.ra_column].values
        self.dec = self.input_data[self.dec_column].values

    def __len__(self):
        return len(self.ra)

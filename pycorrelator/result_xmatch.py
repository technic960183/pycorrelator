from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from .catalog import Catalog

class XMatchResult:

    def __init__(self, cat1: Catalog, cat2: Catalog, tolerance, result_dict: defaultdict):
        self.cat1 = cat1
        self.cat2 = cat2
        self.tolerance = tolerance
        self.result_dict = result_dict
        self.result_dict_reserve = None
    
    def __str__(self):
        return f"XMatchResult of cat1 with {len(self.cat1)} objects and cat2 with {len(self.cat2)} objects."

    def get_result_dict(self) -> defaultdict:
        # [FIXME] Temporary fix for the potential issue of unsorted dictionary
        # This solution impacts the performance of the code.
        renormalization_dd = defaultdict(list)
        for idx in self.cat1.get_indexes():
            renormalization_dd[idx] = self.result_dict[idx]
        return renormalization_dd

    def get_result_dict_reserve(self) -> defaultdict:
        # if self.result_dict_reserve is None: # [TODO] Save the result_dict_reserve to improve performance
        temp_dd = defaultdict(list) # Improve the performance after fixing the issue of unsorted dictionary
        for k, v in self.result_dict.items():
            for vv in v:
                temp_dd[vv].append(k)
        self.result_dict_reserve = defaultdict(list)
        for idx in self.cat2.get_indexes():
            self.result_dict_reserve[idx] = temp_dd[idx]
        return self.result_dict_reserve
    
    def get_dataframe1(self, min_match=0, coord_columns=['Ra', 'Dec'],
                       retain_all_columns=True, retain_columns=None) -> pd.DataFrame:
        '''Get the first catalog with the number of matches as a pandas dataframe.
        
        Parameters
        ----------
        min_match : int, optional
            The minimum number of matches for an object to be included in the dataframe. Default is 0.
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
            The dataframe of the first catalog with the number of matches.
        '''
        idxes_array = self.cat1.get_indexes()
        coords_array = self.cat1.get_coordiantes()
        data_df = pd.DataFrame(coords_array, columns=coord_columns, index=idxes_array)
        data_df['N_match'] = [len(v) for v in self.get_result_dict().values()]
        append_df = self.cat1.get_appending_data(retain_all_columns, retain_columns)
        if len(append_df.columns) > 0:
            data_df = pd.concat([data_df, append_df], axis=1)
        data_df = data_df[data_df['N_match'] >= min_match]
        return data_df
    
    def get_dataframe2(self, min_match=0, coord_columns=['Ra', 'Dec'],
                       retain_all_columns=True, retain_columns=None) -> pd.DataFrame:
        '''Get the second catalog with the number of matches as a pandas dataframe.

        Please refer to the `get_dataframe1()` and replace the 'first catalog' with the 'second catalog'.
        '''
        idxes_array = self.cat2.get_indexes()
        coords_array = self.cat2.get_coordiantes()
        data_df = pd.DataFrame(coords_array, columns=coord_columns, index=idxes_array)
        data_df['N_match'] = [len(v) for v in self.get_result_dict_reserve().values()]
        append_df = self.cat2.get_appending_data(retain_all_columns, retain_columns)
        if len(append_df.columns) > 0:
            data_df = pd.concat([data_df, append_df], axis=1)
        data_df = data_df[data_df['N_match'] >= min_match]
        return data_df

    def get_serial_dataframe(self, min_match=1, reverse=False, coord_columns=['Ra', 'Dec'],
                             retain_all_columns=True, retain_columns=None) -> pd.DataFrame:
        '''Get a pandas dataframe with the information of the matching of the two catalogs in a serial manner.

        Each object from the first catalog with sufficient matches (as defined by min_match) appear first,
        followed by their matched objects from the second catalog.

        Parameters
        ----------
        min_match : int, optional
            The minimum number of matches for an object from the first catalog to be included in the dataframe. Default is 1.
        reverse : bool, optional
            Whether to reverse the order of catalogs (i.e., make the second catalog as the first and vice versa). Default is False.
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
            The serial dataframe of the two catalogs with the number of matches.
        '''
        if reverse: # Create a new XMatchResult object with the reversed result_dict
            reserve_result = self.__class__(self.cat2, self.cat1, self.tolerance, self.get_result_dict_reserve())
            df = reserve_result.get_serial_dataframe(min_match, reverse=False, coord_columns=coord_columns,
                                                     retain_all_columns=retain_all_columns,
                                                     retain_columns=retain_columns)
            df['is_cat1'] = ~df['is_cat1']
            return df
        idxes1 = self.cat1.get_indexes()
        if len(self.cat1) == 0:
            return pd.DataFrame(columns=coord_columns)
        idx_combine = []
        is_df1 = []
        n_match = []
        for id in idxes1:
            id2 = self.result_dict[id]
            if len(id2) < min_match:
                continue
            idx_combine.append(id)
            is_df1.append(True)
            idx_combine.extend(id2)
            is_df1 += [False] * len(id2)
            n_match.append(len(id2))
            n_match += [-1] * len(id2)
        if len(idx_combine) == 0:
            return pd.DataFrame(columns=coord_columns)
        idx_combine = np.array(idx_combine, dtype=np.int64)
        is_df1 = np.array(is_df1)
        n1 = len(self.cat1)
        idx_combine[~is_df1] += n1
        idxes_array1 = self.cat1.get_indexes()
        idxes_array2 = self.cat2.get_indexes()
        df1 = pd.DataFrame(self.cat1.get_coordiantes(), columns=coord_columns, index=idxes_array1)
        df2 = pd.DataFrame(self.cat2.get_coordiantes(), columns=coord_columns, index=idxes_array2)
        append_df1 = self.cat1.get_appending_data(retain_all_columns, retain_columns, invalid_key_error=False)
        append_df2 = self.cat2.get_appending_data(retain_all_columns, retain_columns, invalid_key_error=False)
        if len(append_df1.columns) > 0:
            append_df1.index = idxes_array1
            df1 = pd.concat([df1, append_df1], axis=1)
        if len(append_df2.columns) > 0:
            append_df2.index = idxes_array2
            df2 = pd.concat([df2, append_df2], axis=1)
        combined_df = pd.concat([df1, df2], ignore_index=False)
        data_df = combined_df.iloc[idx_combine]
        if retain_columns is not None:
            non_existent_columns = [col for col in retain_columns if col not in data_df.columns]  
            if non_existent_columns:
                raise KeyError(f"Columns {non_existent_columns} are not in the input DataFrame")
        data_df.insert(2, 'N_match', n_match)
        data_df.insert(3, 'is_cat1', is_df1)
        return data_df
            
    def number_distribution(self) -> Counter:
        """Get the distribution of the number of matches for each object in the first catalog.

        Returns
        -------
        collections.Counter
            The distribution of the number of matches for each object in the first catalog.
        """
        Ns = [len(v) for v in self.get_result_dict().values()]
        unique_counts = Counter(Ns)
        return unique_counts
        
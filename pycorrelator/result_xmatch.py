import csv
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .catalog import Catalog

class XMatchResult:

    def __init__(self, cat1: Catalog, cat2: Catalog, tolerance, result_dict: defaultdict):
        self.cat1 = cat1
        self.cat2 = cat2
        self.tolerance = tolerance
        self.result_dict = result_dict
        self.reverse_dict = None
    
    # def __str__(self):
    #     return f"XMatchResult: number of matches={len(self.result_dict)}"

    def get_result_dict(self):
        return self.result_dict
    
    def get_dataframe1(self, min_match=1, coord_columns=['Ra', 'Dec'],
                       retain_all_columns=True, retain_columns=[]) -> pd.DataFrame:
        '''
        Purpose: Get the dataframe of the first catalog with the number of matches.
        Parameters:
            - min_match (int, optional): The minimum number of matches for an object to be included
                in the dataframe. Default is 1.
            - coord_columns (List[str], optional): The names of the columns for the coordinates.
                Default is ['Ra', 'Dec'].
            - retain_all_columns (bool, optional): Whether to retain all the columns in the
                input (dataframe). Default is True.
            - retain_columns (List[str, optional]): The names of the columns to retain in the output
                dataframe. Will override retain_all_columns if not empty. Default is an empty list.
        Returns:
            - pd.DataFrame: The dataframe of the first catalog with the number of matches.
        '''
        idxes_array = self.cat1.get_indexes()
        coords_array = self.cat1.get_coordiantes()
        data_df = pd.DataFrame(coords_array, columns=coord_columns, index=idxes_array)
        data_df['N_match'] = [len(v) for v in self.result_dict.values()]
        append_df = self.cat1.get_appending_data(retain_all_columns, retain_columns)
        if len(append_df.columns) > 0:
            data_df = pd.concat([data_df, append_df], axis=1)
        data_df = data_df[data_df['N_match'] >= min_match]
        return data_df
    
    def get_dataframe2(self, coord_columns=['Ra', 'Dec'], min_match=1,
                       retain_all_columns=True, retain_columns=[]) -> pd.DataFrame:
        # Consider creating a new XMatchResult object with the reversed result_dict than calling get_dataframe1
        idxes_array = self.cat2.get_indexes()
        coords_array = self.cat2.get_coordiantes()
        data_df = pd.DataFrame(coords_array, columns=coord_columns, index=idxes_array)
        data_df['N_match'] = [len(v) for v in self.result_dict.values()] # [FIXME] Reverse the result_dict
        data_df = data_df[data_df['N_match'] >= min_match]
        return data_df

    def get_serial_dataframe(self, coord_columns=['Ra', 'Dec'], min_match=1, reverse=False,
                             retain_all_columns=True, retain_columns=[]) -> pd.DataFrame:
        # [TODO] To get a reversed dataframe, we need to reverse the result_dict
        # and create a new XMatchResult object with the reversed result_dict.
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
        non_existent_columns = [col for col in retain_columns if col not in data_df.columns]  
        if non_existent_columns:
            raise KeyError(f"Columns {non_existent_columns} are not in the input DataFrame")
        data_df.insert(2, 'N_match', n_match)
        data_df.insert(3, 'is_cat1', is_df1)
        return data_df
    
    @staticmethod
    def load_from_serial_dataframe(df, tolerance=None):
        rtn = XMatchResult(None, None, tolerance, None)
        rtn.df_combine = df
        return rtn

    def save_as_skyviewer(self, pathname, radius=5, colors={4: 'red', 3: 'yellow', 2: '#90EE90'}):
        """
        Purpose: Saves the cross-match result as a skyviewer file.
        Parameters:
            - pathname (str): The pathname of the output file.
            - radius (float): The radius of the circles in the skyviewer in arcsec.
            - colors (dict): The colors of the circles in the skyviewer. The keys are the number of 
            matched objects, and the values are the colors of the circles.
        """
        key_max = max(list(colors.keys()))
        coordinate = lambda k: tuple(self.cat1.iloc[int(k)][['Ra', 'Dec']].values)
        # List of tuples
        data = [coordinate(k) + (radius, colors[key_max]) for k, v in self.result_dict.items() if len(v) >= key_max]
        colors.pop(key_max)
        for num, color in colors.items():
            data += [coordinate(k) + (radius, color) for k, v in self.result_dict.items() if len(v) == num]
        with open(pathname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['RA', 'DEC', 'RADIUS', 'COLOR'])
            writer.writerows(data)
            
    def number_distribution(self):
        Ns = [len(v) for v in self.result_dict.values()]
        unique_counts = Counter(Ns)
        return unique_counts

    def draw_number_distribution(self, start=2, end=5, pathname=None):
        """
        Purpose: Draws the distribution of the number of matches for each object in the first catalog.
        Parameters:
            - start (int): The start of the range of the number of matches (inclusive).
            - end (int): The end of the range of the number of matches (inclusive).
            - pathname (str): The pathname of the output file. If None, the plot will be shown.
        """
        num = lambda n: len([k for k, v in self.result_dict.items() if len(v) == n])
        x_list = [i for i in range(start, end + 1)]
        y_list = [num(i) for i in x_list]
        bars = plt.bar(x_list, y_list)
        plt.yscale('log')
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue  # Skip bars with a height of 0
            # Place the text at the height of the bar, accounting for log scale
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')
        if pathname is not None:
            plt.savefig(pathname)
        else:
            plt.show()
        
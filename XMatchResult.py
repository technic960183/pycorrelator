import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
from ..objects.FileIO import WriteTractor

class XMatchResult:

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, tolerance, result_dict: dict):
        self.df1 = df1
        self.df2 = df2
        self.tolerance = tolerance
        self.result_dict = result_dict
        # self.df_combine = None
    
    def __str__(self):
        return f"XMatchResult: number of matches={len(self.result_dict)}"

    def get_result_dict(self):
        return self.result_dict
    
    def get_dataframe1(self, columns=['Ra', 'Dec', 'brickid', 'objid']):
        idx = np.array(list(self.result_dict.keys())).astype(int)
        data_df = self.df1.iloc[idx][columns]
        data_df['N_match'] = [len(v) for v in self.result_dict.values()]
        return data_df

    def get_serial_dataframe(self, columns=['Ra', 'Dec', 'brickid', 'objid']):
        idx1 = np.array(list(self.result_dict.keys())).astype(int)
        idx_combine = []
        is_df1 = []
        n_match = []
        for id in idx1:
            idx_combine.append(id)
            is_df1.append(True)
            id2 = self.result_dict[id]
            idx_combine += id2
            is_df1 += [False] * len(id2)
            n_match.append(len(id2))
            n_match += [0] * len(id2)
        idx_combine = np.array(idx_combine).astype(int)
        is_df1 = np.array(is_df1)
        n1 = len(self.df1)
        idx_combine[~is_df1] += n1
        combined_df = pd.concat([self.df1, self.df2], ignore_index=True)
        data_df = combined_df.iloc[idx_combine][columns]
        data_df['N_match'] = n_match
        data_df['is_df1'] = is_df1
        return data_df
    
    @staticmethod
    def load_from_serial_dataframe(df, tolerance=None):
        rtn = XMatchResult(None, None, tolerance, None)
        rtn.df_combine = df
        return rtn
    
    def save_as_h5(self, pathname, full=False, columns=['Ra', 'Dec', 'brickid', 'objid']):
        if full:
            WriteTractor(pathname, self.get_serial_dataframe(columns=columns))
        else:
            WriteTractor(pathname, self.get_dataframe1(columns=columns))

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
        coordinate = lambda k: tuple(self.df1.iloc[int(k)][['Ra', 'Dec']].values)
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

    def apply_filters(self, filters, replace=True):
        """
        Purpose: Applies filters to the cross-match result.
        Parameters:
            - filters (list): A list of filters. Each filter is a function that takes in a dictionary
            of the cross-match result and returns a boolean value.
            - replace (bool): Whether to replace the cross-match result with the filtered result.
        Return: The filtered serial_dataframe.
        """
        # [TODO] Generalize for df1 df2's data structure
        # [FIXME] The filterings should be limited to df2
        raise BrokenPipeError("The implementation is broken.")
        filtered_df = self.df_combine.copy()
        if 'N_match_filtered' in filtered_df.columns:
            # filtered_df.drop(columns=['N_match_filtered'], inplace=True)
            print("Warning: N_match_filtered already exists in the DataFrame. The column will be overwritten.")
        for filter in filters:
            filtered_df = filter.apply(filtered_df, copy=True)
        df1_idx = np.where(filtered_df['is_df1'])[0]
        df1_idx_plus = np.append(df1_idx, len(filtered_df))
        N_match_new = np.zeros(len(filtered_df), dtype=np.int64) - 1
        N_match_new[df1_idx] = np.diff(df1_idx_plus) - 1
        filtered_df['N_match_filtered'] = N_match_new
        filtered_df = filtered_df[filtered_df['N_match_filtered'] != 0].copy()
        filtered_df[filtered_df['is_df1'] == False]['N_match_filtered'] = 0
        if replace:
            self.df_combine = filtered_df
        return filtered_df
        
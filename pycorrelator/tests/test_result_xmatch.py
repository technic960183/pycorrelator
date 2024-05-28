import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pycorrelator import xmatch
from pycorrelator.catalog import Catalog

class TestXMatchResult_Methods(unittest.TestCase):

    def setUp(self):

        def generate_offset_groups(base_ra: NDArray, base_dec: NDArray):
            if base_ra.shape[0] != base_dec.shape[0]:
                raise ValueError("The two arrays must have the same length.")
            l = base_ra.shape[0]
            base_coords = np.vstack([base_ra, base_dec]).T
            unit_ra = np.vstack([np.ones(l), np.zeros(l)]).T
            unit_dec = np.vstack([np.zeros(l), np.ones(l)]).T
            coords = np.array([base_coords + 120 * i * unit_ra + 20 * j * unit_dec for i in range(3) for j in range(-1, 2)])
            coords = coords.reshape(-1, coords.shape[-1])
            coords[:,0] = coords[:,0] % 360
            return coords
        
        base1 = np.array([-0.5, 0.1, 0.3]), np.array([0.2, -0.4, 0])
        base2 = np.array([0.5, -0.4]), np.array([-0.1, 0.6])
        self.coords1 = generate_offset_groups(*base1)
        self.coords2 = generate_offset_groups(*base2)
        self.n1 = base1[0].shape[0]
        self.n2 = base2[0].shape[0]

    def test_get_result_dict(self):
        result = xmatch(self.coords1, self.coords2, 2)
        result_dict = result.get_result_dict()
        print(result_dict)
        for i in range(self.coords1.shape[0]):
            self.assertIn(i, result_dict)
            self.assertEqual(len(result_dict[i]), self.n2)
            for j in range(i // self.n1 * self.n2, (i // self.n1 + 1) * self.n2):
                self.assertIn(j, result_dict[i])

    def test_get_dataframe1(self):
        result = xmatch(self.coords1, self.coords2, 2)
        columns = ['Ra', 'Deccc']
        df = result.get_dataframe1(coord_columns=columns)
        self.assertEqual(len(df), self.coords1.shape[0])
        self.assertListEqual(list(df.columns), columns + ['N_match'])
        for i in range(self.coords1.shape[0]):
            self.assertEqual(df.loc[i, 'N_match'], self.n2)
            self.assertAlmostEqual(df.loc[i, columns[0]], self.coords1[i, 0])
            self.assertAlmostEqual(df.loc[i, columns[1]], self.coords1[i, 1])
        self.assertListEqual(list(df.index), list(range(self.coords1.shape[0])))

    def test_get_dataframe1_min_match(self):
        result = xmatch(self.coords1, self.coords2, 2)
        df = result.get_dataframe1(min_match=self.n2 + 1)
        self.assertEqual(len(df), 0)

    def test_get_dataframe1_retain_all_columns_True(self):
        columns = ['RA', 'DEC']
        retain_columns = ['A', 'B']
        df1 = pd.DataFrame(self.coords1, columns=columns)
        df1['A'] = np.cos(self.coords1[:, 0]) + np.arange(self.coords1.shape[0])
        df1['B'] = np.sin(self.coords1[:, 1]) + np.arange(self.coords1.shape[0])
        result = xmatch(df1, self.coords2, 2)
        df = result.get_dataframe1(coord_columns=columns, retain_all_columns=True)
        self.assertEqual(len(df), self.coords1.shape[0])
        self.assertListEqual(list(df.columns), columns + ['N_match'] + retain_columns)
        for i in range(self.coords1.shape[0]):
            for rc in retain_columns:
                self.assertAlmostEqual(df.loc[i, rc], df1.loc[i, rc])
        self.assertListEqual(list(df.index), list(range(self.coords1.shape[0])))

    def test_get_dataframe1_retain_all_columns_False(self):
        columns = ['RA', 'DEC']
        retain_columns = ['A', 'B']
        df1 = pd.DataFrame(self.coords1, columns=columns)
        df1['A'] = np.cos(self.coords1[:, 0]) + np.arange(self.coords1.shape[0])
        df1['B'] = np.sin(self.coords1[:, 1]) + np.arange(self.coords1.shape[0])
        result = xmatch(df1, self.coords2, 2)
        df = result.get_dataframe1(coord_columns=columns, retain_all_columns=False)
        self.assertEqual(len(df), self.coords1.shape[0])
        self.assertListEqual(list(df.columns), columns + ['N_match'])

    def test_get_dataframe1_retain_columns(self):
        columns = ['RA', 'DEC']
        retain_columns = ['A', 'B']
        df1 = pd.DataFrame(self.coords1, columns=columns)
        df1['A'] = np.cos(self.coords1[:, 0]) + np.arange(self.coords1.shape[0])
        df1['B'] = np.sin(self.coords1[:, 1]) + np.arange(self.coords1.shape[0])
        result = xmatch(df1, self.coords2, 2)
        df = result.get_dataframe1(coord_columns=columns, retain_columns=['A'])
        self.assertEqual(len(df), self.coords1.shape[0])
        self.assertListEqual(list(df.columns), columns + ['N_match'] + ['A'])
        for i in range(self.coords1.shape[0]):
            self.assertAlmostEqual(df.loc[i, 'A'], df1.loc[i, 'A'])
        self.assertListEqual(list(df.index), list(range(self.coords1.shape[0])))

    def test_get_dataframe2(self):
        result = xmatch(self.coords1, self.coords2, 2)
        columns = ['Ra', 'Deccc']
        df = result.get_dataframe2(coord_columns=columns)
        self.assertEqual(len(df), self.coords2.shape[0])
        self.assertListEqual(list(df.columns), columns + ['N_match'])
        for i in range(self.coords2.shape[0]):
            self.assertEqual(df.loc[i, 'N_match'], self.n1)
            self.assertAlmostEqual(df.loc[i, columns[0]], self.coords2[i, 0])
            self.assertAlmostEqual(df.loc[i, columns[1]], self.coords2[i, 1])
        self.assertListEqual(list(df.index), list(range(self.coords2.shape[0])))

    def test_get_serial_dataframe(self):
        result = xmatch(self.coords1, self.coords2, 2)
        columns = ['Ra', 'Deccc']
        df = result.get_serial_dataframe(coord_columns=columns)
        self.assertEqual(len(df), self.coords1.shape[0] + self.coords2.shape[0] * self.n1)
        self.assertListEqual(list(df.columns), columns + ['N_match', 'is_cat1'])
        for i in range(self.coords1.shape[0]):
            idx = i * (self.n2 + 1)
            self.assertEqual(df.iloc[idx]['N_match'], self.n2)
            self.assertEqual(df.iloc[idx]['is_cat1'], True)
            self.assertAlmostEqual(df.iloc[idx][columns[0]], self.coords1[i, 0])
            self.assertAlmostEqual(df.iloc[idx][columns[1]], self.coords1[i, 1])
            for j in range(self.n2):
                idx = i * (self.n2 + 1) + j + 1
                self.assertEqual(df.iloc[idx]['N_match'], -1)
                self.assertEqual(df.iloc[idx]['is_cat1'], False)
                self.assertAlmostEqual(df.iloc[idx][columns[0]], self.coords2[i // self.n1 * self.n2 + j, 0])
                self.assertAlmostEqual(df.iloc[idx][columns[1]], self.coords2[i // self.n1 * self.n2 + j, 1])

    def test_get_serial_dataframe_reverse(self):
        result = xmatch(self.coords1, self.coords2, 2)
        columns = ['Ra', 'Deccc']
        df = result.get_serial_dataframe(coord_columns=columns, reverse=True)
        self.assertEqual(len(df), self.coords2.shape[0] + self.coords1.shape[0] * self.n2)
        self.assertListEqual(list(df.columns), columns + ['N_match', 'is_cat1'])
        for i in range(self.coords2.shape[0]):
            idx = i * (self.n1 + 1)
            self.assertEqual(df.iloc[idx]['N_match'], self.n1)
            self.assertEqual(df.iloc[idx]['is_cat1'], False)
            self.assertAlmostEqual(df.iloc[idx][columns[0]], self.coords2[i, 0])
            self.assertAlmostEqual(df.iloc[idx][columns[1]], self.coords2[i, 1])
            for j in range(self.n1):
                idx = i * (self.n1 + 1) + j + 1
                self.assertEqual(df.iloc[idx]['N_match'], -1)
                self.assertEqual(df.iloc[idx]['is_cat1'], True)
                self.assertAlmostEqual(df.iloc[idx][columns[0]], self.coords1[i // self.n2 * self.n1 + j, 0])
                self.assertAlmostEqual(df.iloc[idx][columns[1]], self.coords1[i // self.n2 * self.n1 + j, 1])

    def test_get_serial_dataframe_min_match(self):
        result = xmatch(self.coords1, self.coords2, 2)
        df = result.get_serial_dataframe(min_match=self.n2 + 1)
        self.assertEqual(len(df), 0)

    def test_get_serial_dataframe_retain_all_columns(self):
        columns = ['RA', 'DEC']
        retain_columns = ['A', 'B']
        df1 = pd.DataFrame(self.coords1, columns=columns)
        df1['A'] = np.cos(self.coords1[:, 0]) + np.arange(self.coords1.shape[0])
        df1['B'] = np.sin(self.coords1[:, 1]) + np.arange(self.coords1.shape[0])
        result = xmatch(df1, self.coords2, 2)
        df = result.get_serial_dataframe(coord_columns=columns, retain_all_columns=True)
        self.assertEqual(len(df), self.coords1.shape[0] + self.coords2.shape[0] * self.n1)
        self.assertListEqual(list(df.columns), columns + ['N_match', 'is_cat1'] + retain_columns)
        for i in range(self.coords1.shape[0]):
            idx = i * (self.n2 + 1)
            for rc in retain_columns:
                self.assertAlmostEqual(df.iloc[idx][rc], df1.loc[i, rc])
            for j in range(self.n2):
                idx = i * (self.n2 + 1) + j + 1
                for rc in retain_columns:
                    value = df.iloc[idx][rc]
                    self.assertTrue(value is None or pd.isna(value))

    def test_get_serial_dataframe_retain_columns(self):
        columns = ['RA', 'DEC']
        retain_columns = ['A', 'C']
        df1 = pd.DataFrame(self.coords1, columns=columns)
        df1['A'] = np.cos(self.coords1[:, 0]) + np.arange(self.coords1.shape[0])
        df1['B'] = np.sin(self.coords1[:, 1]) + np.arange(self.coords1.shape[0])
        df2 = pd.DataFrame(self.coords2, columns=columns)
        df2['A'] = -np.sin(self.coords2[:, 0]) - np.arange(self.coords2.shape[0])
        df2['C'] = np.abs(self.coords2[:, 1]) + np.arange(self.coords2.shape[0])
        result = xmatch(df1, df2, 2)
        df = result.get_serial_dataframe(coord_columns=columns, retain_columns=retain_columns)
        self.assertEqual(len(df), self.coords1.shape[0] + self.coords2.shape[0] * self.n1)
        self.assertListEqual(list(df.columns), columns + ['N_match', 'is_cat1'] + retain_columns)
        for i in range(self.coords1.shape[0]):
            idx = i * (self.n2 + 1)
            self.assertAlmostEqual(df.iloc[idx]['A'], df1.loc[i, 'A'])
            value = df.iloc[idx]['C']
            self.assertTrue(value is None or pd.isna(value))
            for j in range(self.n2):
                idx = i * (self.n2 + 1) + j + 1
                self.assertAlmostEqual(df.iloc[idx]['A'], df2.loc[i // self.n1 * self.n2 + j, 'A'])
                self.assertAlmostEqual(df.iloc[idx]['C'], df2.loc[i // self.n1 * self.n2 + j, 'C'])
    
    @unittest.skip("Future functionality")
    def test_get_multiindex_dataframe(self):
        result = xmatch(self.coords1, self.coords2, 2)
        columns = ['Ra', 'Deccc']
        df = result.get_multiindex_dataframe(coord_columns=columns)
        self.assertEqual(len(df), self.coords1.shape[0] + self.coords2.shape[0] * self.n1)
        self.assertListEqual(list(df.columns), ['Catalog', 'Column'])
        self.assertListEqual(list(df.index), ['Group', 'Object'])
        sizes = df.groupby('Group').size()
        self.assertEqual(len(sizes), self.coords1.shape[0])
        self.assertTrue(all(sizes == self.n2))
        for i in range(self.coords1.shape[0]):
            group_df = df.loc[i]
            continue # [TODO] Check the content of the group dataframe

    # [FIXME] Write a test to check that if itterating over the deaultdict, the keys won't be in the correct order.
    # Thus yielding an incorrect result of N_match.

    
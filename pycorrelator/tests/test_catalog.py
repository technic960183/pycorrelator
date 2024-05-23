import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import numpy as np
import pandas as pd
from pycorrelator.catalog import Catalog
from pycorrelator.utilities_spherical import generate_random_point

class TestCatalog_RandomCheckInputOutput(unittest.TestCase):

    @staticmethod
    def get_input_output_pair(N):
        ra, dec = generate_random_point(N)
        expected_output = np.vstack([ra, dec]).T
        return (ra, dec), expected_output

    def test_random_np(self):
        for i in range(10):
            (ra, dec), expected_output = self.get_input_output_pair(N=1000)
            catalog = Catalog(np.vstack([ra, dec]).T)
            code_output = catalog.get_coordiantes()
            self.assertEqual(code_output.tolist(), expected_output.tolist())
    
    def test_random_pd(self):
        for i in range(10):
            (ra, dec), expected_output = self.get_input_output_pair(N=1000)
            catalog = Catalog(pd.DataFrame({'Ra': ra, 'Dec': dec}))
            code_output = catalog.get_coordiantes()
            self.assertEqual(code_output.tolist(), expected_output.tolist())

class TestCatalog_ValidInput(unittest.TestCase):

    def setUp(self):
        self.ra = np.array([10, 20, 30, 40])
        self.dec = np.array([-10, -20, -30, -40])
        self.expected_output = np.vstack([self.ra, self.dec]).T
        self.parameter = None
        self.code_output = None

    def test_np(self):
        self.parameter = np.vstack([self.ra, self.dec]).T

    def test_pd(self):
        self.parameter = pd.DataFrame({'Ra': self.ra, 'Dec': self.dec})

    def test_pd_column_name_lower(self):
        self.parameter = pd.DataFrame({'ra': self.ra, 'dec': self.dec})

    def test_pd_column_name_upper(self):
        self.parameter = pd.DataFrame({'RA': self.ra, 'DEC': self.dec})

    def test_pd_column_name_mixed(self):
        self.parameter = pd.DataFrame({'RA': self.ra, 'dec': self.dec})

    def tearDown(self):
        catalog = Catalog(self.parameter)
        code_output = catalog.get_coordiantes()
        self.assertEqual(code_output.tolist(), self.expected_output.tolist())

class TestCatalog_InvalidInput(unittest.TestCase):

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            Catalog("invalid input type")

    def test_np_shape_1d(self):
        with self.assertRaises(ValueError):
            Catalog(np.array([1, 2, 3]))  # 1D array

    def test_np_shape_2x3(self):        
        with self.assertRaises(ValueError):
            Catalog(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_pd_column_missing_dec(self):
        with self.assertRaises(ValueError):
            Catalog(pd.DataFrame({'Ra': [1, 2, 3], 'Dog': [1, 2, 3]}))

    def test_pd_column_missing_ra(self):
        with self.assertRaises(ValueError):
            Catalog(pd.DataFrame({'Roy': [1, 2, 3], 'Dec': [1, 2, 3]}))

    def test_np_contains_inf(self):
        with self.assertRaises(ValueError):
            Catalog(np.array([[1, 2], [3, 4], [5, np.inf]]))

    def test_pd_contains_nan(self):
        with self.assertRaises(ValueError):
            Catalog(pd.DataFrame({'Ra': [1, 2, 3], 'Dec': [1, 2, np.nan]}))


if __name__ == '__main__':
    unittest.main()
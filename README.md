# pycorrelator
A Python package for cross correlation and self correlation in spherical coordinates.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)

## Introduction
`pycorrelator` is a Python package designed to perform cross correlation and self correlation analyses in spherical coordinates. This is particularly useful in fields such as astrophysics, geophysics, and any domain where objects are naturally distributed on a spherical surface.
Currently, this package only support astrophysics coordinates `(Ra, Dec)` in degrees. More units and naming convention will be supported in the future.

## Features
- Efficient computation ( $O(N\log N)$ ) of cross correlation in spherical coordinates.
- Friends-of-Friends (FoF) analysis in spherical coordinates.
- Easy integration with existing data processing packages, such as `pandas`.

## Installation
You can install `pycorrelator` by cloning the codes:
```bash
git clone https://github.com/technic960183/pycorrelator.git
```

pip install will be supported in the future.

## Usage
There are two main functions in `pycorrelator`:
- `xmatch`: Cross match two sets of objects in spherical coordinates.
- `group_by_quadtree`: Group objects by quadtree algorithm.

### xmatch
`xmatch(catalog1, catalog2, tolerance, verbose=True) -> XMatchResult`

The `xmatch` function performs a cross-match between two catalogs of objects in spherical coordinates.

##### Parameters

- `catalog1` (array-like): The first catalog. It can be either a numpy array or a pandas dataframe.
    * np.array: The array must have a shape of (N, 2), representing N points with
    two values: [ra (azimuth, longitude), dec (alltitude, latitude)].
    * pandas.DataFrame: The dataframe must have two columns named 'Ra' and 'Dec' (or all the
    possible combinations with 'ra', 'dec'; 'RA', 'DEC').

- `catalog2` (array-like): The second catalog. This should have the same format as `catalog1`.

- `tolerance` (float): The tolerance for the cross-match in degrees. Two objects from `catalog1` and `catalog2` are considered to be a match if their angular separation is within this tolerance.

- `verbose` (bool, optional): Whether to print the progress of the cross-match. If `True`, the function will print messages indicating the progress of the cross-match. Default is `True`.

##### Returns

- `XMatchResult`: A `XMatchResult` object that contains the cross-match result. See [XMatchResult](#xmatchresult) for more details.

### XMatchResult
The `XMatchResult` object contains the result of `xmatch` function. It has the following methods:

#### get_dataframe1()
`get_dataframe1(min_match=0, coord_columns=['Ra', 'Dec'], retain_all_columns=True, retain_columns=None) -> pandas.DataFrame`

Get the dataframe of the first catalog with the number of matches.

##### Parameters

- `min_match` (int, optional): The minimum number of matches required for an object in the first catalog to be included. Default is `0`.

- `coord_columns` (list of str, optional): The names of the columns representing the coordinates in the dataframe. Default is `['Ra', 'Dec']`.

- `retain_all_columns` (bool, optional): Whether to retain all the columns in the input (dataframe). Default is `True`.

- `retain_columns` (list of str, optional): The names of the columns to retain in the output dataframe. Will override `retain_all_columns` if not empty. Default is `None`.

##### Returns

- `pandas.DataFrame`: A pandas dataframe containing the coordinates of the objects in the first catalog with the number of matches.

#### get_dataframe2()
`get_dataframe2(min_match=0, coord_columns=['Ra', 'Dec'], retain_all_columns=True, retain_columns=None) -> pandas.DataFrame`

Get the dataframe of the second catalog with the number of matches. Please refer to [`get_dataframe1()`](#get_dataframe1) for the parameters and return value.

#### get_serial_dataframe()
`get_serial_dataframe(min_match=1, reverse=False, coord_columns=['Ra', 'Dec'], retain_all_columns=True, retain_columns=None) -> pandas.DataFrame`

This function combines the two catalogs based on the number of matches. Each object from the first catalog with sufficient matches (as defined by `min_match`) appears first, followed by their matched objects from the second catalog.

##### Parameters

- `min_match` (int, optional): The minimum number of matches for an object from the first catalog to be included in the dataframe. Default is `1`.

- `reverse` (bool, optional): Whether to reverse the order of catalogs (i.e., make the second catalog as the first and vice versa). Default is `False`.

- `coord_columns` (List[str], optional): The names of the columns for the coordinates. Default is `['Ra', 'Dec']`.

- `retain_all_columns` (bool, optional): Whether to retain all the columns in the input (dataframe). Default is `True`.

- `retain_columns` (List[str], optional): The names of the columns to retain in the output dataframe. Will override `retain_all_columns` if not empty. Default is `None`.

##### Returns

- `pandas.DataFrame`: The serial dataframe of the two catalogs with the number of matches.

### group_by_quadtree
`group_by_quadtree(catalog, tolerance) -> FoFResult`

The `group_by_quadtree` function groups objects in a catalog using a friends-of-friends (FoF) algorithm.

##### Parameters

- `catalog` (array-like): The first catalog. It can be either a numpy array or a pandas dataframe.
    * np.array: The array must have a shape of (N, 2), representing N points with
    two values: [ra (azimuth, longitude), dec (alltitude, latitude)].
    * pandas.DataFrame: The dataframe must have two columns named 'Ra' and 'Dec' (or all the
    possible combinations with 'ra', 'dec'; 'RA', 'DEC').

- `tolerance` (float): The tolerance for grouping in degrees. Two objects are considered to be in the same group if their angular separation is within this tolerance.

##### Returns

- `FoFResult`: A `FoFResult` object that contains the grouping result. See [FoFResult](#fofresult) for more details.

### FoFResult
The `FoFResult` object contains the result of `group_by_quadtree` function. It has the following methods:

#### get_coordinates()
`get_coordinates() -> List[List[Tuple[float, float]]]`

Get the coordinates of the objects in each group. Returns a list of lists, where each inner list contains the coordinates of the objects in a group.

#### get_group_coordinates()
`get_group_coordinates() -> List[Tuple[float, float]]`

Get the coordinates of the group centers. Returns a list of tuples, where each tuple contains the coordinates of the center of a group. The center is defined as the average of the coordinates of the objects in the group.

#### get_group_dataframe()
`get_group_dataframe(min_group_size=1, coord_columns=['Ra', 'Dec'], retain_all_columns=True, retain_columns=None) -> pandas.DataFrame`

Get a dataframe with MultiIndex. The first level of the index is the group index, and the second level is the index from the original catalog.

##### Parameters

- `min_group_size` (int, optional): The minimum number of objects required for a group to be included in the dataframe. Default is `1` (i.e., all groups are included).

- `coord_columns` (list of str, optional): The names of the columns representing the coordinates in the dataframe. Default is `['Ra', 'Dec']`.

- `retain_all_columns` (bool, optional): Whether to retain all the columns in the input (dataframe). Default is `True`.

- `retain_columns` (list of str, optional): The names of the columns to retain in the output dataframe. Will override `retain_all_columns` if not empty. Default is `None`.

##### Returns

- `pandas.DataFrame`: A pandas dataframe with MultiIndex, where the first level is the group index and the second level is the index from the original catalog.

## Examples

### Cross-matching two catalogs
```python
import numpy as np
from pycorrelator import xmatch

# Create two mock catalogs
catalog1 = np.array([[80.894, 41.269], [120.689, -41.269], [10.689, -41.269]])
catalog2 = np.array([[10.688, -41.270], [10.689, -41.270], [10.690, -41.269], [120.690, -41.270]])

# Perform the cross-matching
result = xmatch(catalog1, catalog2, tolerance=0.01)

# Get the result
print(result.get_serial_dataframe(min_match=0))
```

Expected output:
```
        Ra     Dec  N_match  is_cat1
0   80.894  41.269        0     True
1  120.689 -41.269        1     True
3  120.690 -41.270       -1    False
2   10.689 -41.269        3     True
0   10.688 -41.270       -1    False
1   10.689 -41.270       -1    False
2   10.690 -41.269       -1    False
```

### Clustering using FoF algorithm
```python
import pandas as pd
from pycorrelator import group_by_quadtree

# Create a mock catalog
catalog = pd.DataFrame([[80.894, 41.269], [120.689, -41.269], 
                        [10.689, -41.269], [10.688, -41.270], 
                        [10.689, -41.270], [10.690, -41.269], 
                        [120.690, -41.270]], columns=['ra', 'dec'])

# Perform the clustering
result = group_by_quadtree(catalog, tolerance=0.01)

# Get the result
print(result.get_coordinates())
```

Expected output:
``` 
[[(80.894, 41.269)],
 [(120.689, -41.269), (120.69, -41.27)],
 [(10.689, -41.269), (10.688, -41.27), (10.689, -41.27), (10.69, -41.269)]]
```

## Contributing
If you find any bugs or potential issues, please report it directly to me (via Slack or E-mail), or start an [issue](https://github.com/technic960183/pycorrelator/issues).

If you have any suggestions or feature requests, feel free to start an [issue](https://github.com/technic960183/pycorrelator/issues).

## Citation
If you find `pycorrelator` useful in your research, please consider citing it. Currently, we do not plan to publish a method paper for this package in the year of 2024. However, you can still cite this repository directly.

To cite pycorrelator in your publication, please use the following BibTeX entry:
```bibtex
@misc{pycorrelator,
  author = {Yuan-Ming Hsu},
  title = {pycorrelator: A Python package for cross correlation and self correlation in spherical coordinates.},
  year = {2024},
  howpublished = {\url{https://github.com/technic960183/pycorrelator}},
  note = {Accessed: YYYY-MM}
}
```
Addtionally, you may add a reference to `https://github.com/technic960183/pycorrelator` in the footnote if suitable.

If you publish a paper that uses `pycorrelator`, please let me know. I would be happy to know how this package has been used in research.

# pycorrelator
A Python package for cross correlation and self correlation in spherical coordinates.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction
`pycorrelator` is a Python package designed to perform cross correlation and self correlation analyses in spherical coordinates. This is particularly useful in fields such as astrophysics, geophysics, and any domain where data points are naturally distributed on a spherical surface.
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
- `xmatch`: Cross match two sets of data points in spherical coordinates.
- `group_by_quadtree`: Group data points by quadtree algorithm.

### xmatch
`xmatch(catalog1, catalog2, tolerance, verbose=True) -> XMatchResult`

The `xmatch` function performs a cross-match between two catalogs of data points in spherical coordinates.

#### Parameters

- `catalog1` (array-like): The first catalog. It can be either a numpy array or a pandas dataframe.
    * np.array: The array must have a shape of (N, 2), representing N points with
    two values: [ra (azimuth, longitude), dec (alltitude, latitude)].
    * pandas.DataFrame: The dataframe must have two columns named 'Ra' and 'Dec' (or all the
    possible combinations with 'ra', 'dec'; 'RA', 'DEC').

- `catalog2` (array-like): The second catalog. This should have the same format as `catalog1`.

- `tolerance` (float): The tolerance for the cross-match in degrees. Two data points from `catalog1` and `catalog2` are considered to be a match if their angular separation is within this tolerance.

- `verbose` (bool, optional): Whether to print the progress of the cross-match. If `True`, the function will print messages indicating the progress of the cross-match. Default is `True`.

#### Returns

- `XMatchResult`: A `XMatchResult` object that contains the cross-match result. See [XMatchResult](#xmatchresult) for more details.

### XMatchResult
The `XMatchResult` object contains the result of `xmatch` function. It has the following methods:
[TODO] Add the methods of XMatchResult here.

### group_by_quadtree
`group_by_quadtree(catalog, tolerance) -> FoFResult`

The `group_by_quadtree` function groups data points in a catalog using a friends-of-friends (FoF) algorithm.

#### Parameters

- `catalog` (array-like): The first catalog. It can be either a numpy array or a pandas dataframe.
    * np.array: The array must have a shape of (N, 2), representing N points with
    two values: [ra (azimuth, longitude), dec (alltitude, latitude)].
    * pandas.DataFrame: The dataframe must have two columns named 'Ra' and 'Dec' (or all the
    possible combinations with 'ra', 'dec'; 'RA', 'DEC').

- `tolerance` (float): The tolerance for grouping in degrees. Two data points are considered to be in the same group if their angular separation is within this tolerance.

#### Returns

- `FoFResult`: A `FoFResult` object that contains the grouping result. See [FoFResult](#fofresult) for more details.

### FoFResult
The `FoFResult` object contains the result of `group_by_quadtree` function. It has the following methods:
[TODO] Add the methods of FoFResult here.

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
catalog = pd.DataFrame([[80.894, 41.269], [120.689, -41.269], [10.689, -41.269], [10.688, -41.270], [10.689, -41.270], [10.690, -41.269], [120.690, -41.270]], columns=['ra', 'dec'])

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

## License
Not selected yet. All rights reserved.
[TODO] Selected a license before public.

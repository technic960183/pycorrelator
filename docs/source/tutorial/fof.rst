Example: Clustering using FoF algorithm
=======================================

A friend of friends (FoF) algorithm is useful when you want to find groups of objects that are close
to each other forming clusters. Here is an example of how to perform clustering using the **pycorrelator** package.

First, let's create a mock catalog:

.. code-block:: python

    import pandas as pd

    # Create a mock catalog as a pandas DataFrame
    catalog = pd.DataFrame([[80.894, 41.269, 15.5], [120.689, -41.269, 12.3], 
                            [10.689, -41.269, 18.7], [10.688, -41.270, 14.1], 
                            [10.689, -41.270, 16.4], [10.690, -41.269, 13.2], 
                            [120.690, -41.270, 17.8]], columns=['ra', 'dec', 'mag'])

.. note::
    If you want to use a format other than a pandas DataFrame,
    see the :doc:`supported formats <input_validation>` for more information.

group_by_quadtree()
-------------------

Then, we can perform clustering using the FoF algorithm with the tolerance of 0.01 degree using the
:func:`pycorrelator.group_by_quadtree` function.

.. code-block:: python

    from pycorrelator import group_by_quadtree
    result_object = group_by_quadtree(catalog, tolerance=0.01)

The result object contains the clustering results. Four methods are available to get the results in different formats:

get_group_dataframe()
---------------------

To get the clustering results with the appendind data (``"mag"`` in this case), use the
:func:`pycorrelator.FoFResult.get_group_dataframe` method:

.. code-block:: python

    groups_df = result_object.get_group_dataframe()
    print(groups_df)

Expected output::

                       Ra     Dec   mag
    Group Object                       
    0     0        80.894  41.269  15.5
    1     1       120.689 -41.269  12.3
          6       120.690 -41.270  17.8
    2     2        10.689 -41.269  18.7
          3        10.688 -41.270  14.1
          4        10.689 -41.270  16.4
          5        10.690 -41.269  13.2

This method returns a pandas DataFrame with two layers of indices: the group index and the object index from the original catalog.

You can iterate through each group by:

.. code-block:: python

    for group_index, group in groups_df.groupby('Group'):
        print(f"Print group {group_index}:")
        print(f"The type of group is {type(group)}.")
        print(group, end="\n\n")

Expected output::

    Print group 0:
    The type of group is <class 'pandas.core.frame.DataFrame'>.
                      Ra     Dec   mag
    Group Object                      
    0     0       80.894  41.269  15.5

    Print group 1:
    The type of group is <class 'pandas.core.frame.DataFrame'>.
                       Ra     Dec   mag
    Group Object                       
    1     1       120.689 -41.269  12.3
          6       120.690 -41.270  17.8

    Print group 2:
    The type of group is <class 'pandas.core.frame.DataFrame'>.
                      Ra     Dec   mag
    Group Object                      
    2     2       10.689 -41.269  18.7
          3       10.688 -41.270  14.1
          4       10.689 -41.270  16.4
          5       10.690 -41.269  13.2

Each group is also a pandas DataFrame.

.. note::
    The iterater from ``groupby()`` is extremely slow for large datasets. The current solution is to flatten the
    DataFrame into a single layer of index and manupulate the index directly, or even turn the DataFrame into a numpy array.

If you want DataFrame with a single layer of index and the size of each group as a column, you can use the following code:

.. code-block:: python

    groups_df['group_size'] = groups_df.groupby(level='Group')['Ra'].transform('size')
    groups_df.reset_index(level='Group', inplace=True)
    print(groups_df)

Expected output::

            Group       Ra     Dec   mag  group_size
    Object                                          
    0           0   80.894  41.269  15.5           1
    1           1  120.689 -41.269  12.3           2
    6           1  120.690 -41.270  17.8           2
    2           2   10.689 -41.269  18.7           4
    3           2   10.688 -41.270  14.1           4
    4           2   10.689 -41.270  16.4           4
    5           2   10.690 -41.269  13.2           4

get_group_sizes()
-----------------

To get the size of each group in the order of the group index, use the :func:`pycorrelator.FoFResult.get_group_sizes` method:

.. code-block:: python

    print(result_object.get_group_sizes())

Expected output::

    [1, 2, 4]

get_coordinates()
-----------------

To get the coordinates of the objects in each group, use the :func:`pycorrelator.FoFResult.get_coordinates` method:

.. code-block:: python

    print(result_object.get_coordinates())

Expected output::

    [[(80.894, 41.269)],
     [(120.689, -41.269), (120.69, -41.27)],
     [(10.689, -41.269), (10.688, -41.27), (10.689, -41.27), (10.69, -41.269)]]

get_group_coordinates()
-----------------------

To get the center coordinates of each group, use the :func:`pycorrelator.FoFResult.get_group_coordinates` method:

.. code-block:: python

    print(result_object.get_group_coordinates())

Expected output::

    [(80.894, 41.269), (120.6895, -41.2695), (10.689 , -41.2695)]

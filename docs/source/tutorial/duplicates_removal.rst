Example: Duplicates removal
===========================

One common usage of the FoF algorithm is to remove duplicates from a catalog.
In this example, we will show how to remove duplicates from a catalog using the
:func:`pycorrelator.group_by_quadtree` function.

First, let's create a mock catalog with duplicates:

.. code-block:: python

    import pandas as pd

    # Create a mock catalog as a pandas DataFrame
    catalog = pd.DataFrame([[80.894, 41.269, 1200], [120.689, -41.269, 1500], 
                            [10.689, -41.269, 3600], [10.688, -41.270, 300], 
                            [10.689, -41.270, 1800], [10.690, -41.269, 2400], 
                            [120.690, -41.270, 900], [10.689, -41.269, 2700]], 
                            columns=['ra', 'dec', 'exp_time'])

Here, we actually only have 3 unique objects, but the catalog contains 8 entries and 5 of them are duplicates.

Now we wish to remove the duplicates from the catalog and retain only the unique objects with the highest exposure time.
Here is how we can do it:

.. code-block:: python

    ranking_col = 'exp_time'
    tolerance = 0.01

    from pycorrelator import group_by_quadtree
    result_object = group_by_quadtree(catalog, tolerance=tolerance)
    catalog = result_object.get_group_dataframe()

    catalog['dup_num'] = catalog.groupby('Group')['Ra'].transform('size')
    catalog['rank'] = catalog.groupby('Group')[ranking_col].rank(ascending=False, method='first')
    catalog['rank'] = catalog['rank'].astype(int)
    print(catalog)

Expected output::

                       Ra     Dec  exp_time  dup_num  rank
    Group Object                                          
    0     0        80.894  41.269      1200        1     1
    1     1       120.689 -41.269      1500        2     1
          6       120.690 -41.270       900        2     2
    2     2        10.689 -41.269      3600        5     1
          3        10.688 -41.270       300        5     5
          4        10.689 -41.270      1800        5     4
          5        10.690 -41.269      2400        5     3
          7        10.689 -41.269      2700        5     2

Here I set the tolerance to 0.01, which means that objects with a separation less than 0.01 degrees to any other
object in the same 'cluster' will be considered as duplicates. You need to adjust this value according to the
properties of your catalog. The ``'dup_num'`` column shows the number of duplicates in each group, and the
``'rank'`` column shows the order of the object in the group sorted by the ranking column.

.. note::
    When there are two 'unique' objects that are very close to each other, it is possible that they will be grouped together.
    In an exetrema case, it is possible that a chain of unique objects will be grouped together, linking by their duplicates.
    But this is rare for most catalogs. To solve this problem, you can try to decrease the tolerance value. However, if
    decreasing the tolerance value separates objects that should be considered as duplicates, this package does not provide
    a solution for now. You may need to remove the duplicates manually for those close objects.
    We are now working on some new features related to this issue.

Finally, we can remove the duplicates from the catalog by retaining only the objects with ``'rank'`` equal to 1:

.. code-block:: python

    catalog_no_duplicates = catalog[catalog['rank'] == 1].copy()
    catalog_no_duplicates.drop(columns=['rank'], inplace=True)
    catalog_no_duplicates.reset_index(level='Object', inplace=True)
    print(catalog_no_duplicates)

Expected output::

           Object       Ra     Dec  exp_time  dup_num
    Group                                            
    0           0   80.894  41.269      1200        1
    1           1  120.689 -41.269      1500        2
    2           2   10.689 -41.269      3600        5

Now the catalog contains only the unique objects with the highest exposure time.

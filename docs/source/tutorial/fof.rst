Example: Clustering using FoF algorithm
=======================================

.. code-block:: python

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

Expected output::

    [[(80.894, 41.269)],
     [(120.689, -41.269), (120.69, -41.27)],
     [(10.689, -41.269), (10.688, -41.27), (10.689, -41.27), (10.69, -41.269)]]


Cross-matching two catalogs
---------------------------

.. code-block:: python

    import numpy as np
    from pycorrelator import xmatch

    # Create two mock catalogs
    catalog1 = np.array([[80.894, 41.269], [120.689, -41.269], [10.689, -41.269]])
    catalog2 = np.array([[10.688, -41.270], [10.689, -41.270], [10.690, -41.269], [120.690, -41.270]])

    # Perform the cross-matching
    result = xmatch(catalog1, catalog2, tolerance=0.01)

    # Get the result
    print(result.get_serial_dataframe(min_match=0))

Expected output::

        Ra     Dec      N_match  is_cat1
    0   80.894  41.269        0     True
    1  120.689 -41.269        1     True
    3  120.690 -41.270       -1    False
    2   10.689 -41.269        3     True
    0   10.688 -41.270       -1    False
    1   10.689 -41.270       -1    False
    2   10.690 -41.269       -1    False
Example: Cross-matching two catalogs
====================================

Here is the situation: You have two catalogs A and B, and you want to find the objects in A that are also in B.
However, the coordinates in the two catalogs are not exactly the same, so you need to allow for some tolerance in the matching.

First, let's create two mock catalogs A and B:

.. code-block:: python

    import numpy as np

    # Create two mock catalogs as numpy arrays
    catalogA = np.array([[80.894, 41.269], [120.689, -41.269], [10.689, -41.269]])
    catalogB = np.array([[10.688, -41.270], [10.689, -41.270], [10.690, -41.269], [120.690, -41.270]])

.. note::
    If you want to use a format other than a numpy array, see the :doc:`supported formats <input_validation>` for more information.

xmatch()
--------

Then, we can perform the cross-matching with the tolerance of 0.01 degree using the :func:`pycorrelator.xmatch` function.

.. code-block:: python
    
    from pycorrelator import xmatch
    result_object = xmatch(catalogA, catalogB, tolerance=0.01)

The result object contains the matching results. Three methods are available to get the results in different formats:

get_dataframe1()
----------------

To get the matching results of catalog A, use the :func:`pycorrelator.XMatchResult.get_dataframe1` method:

.. code-block:: python

    print(result_object.get_dataframe1())

Expected output::
    
            Ra     Dec  N_match
    0   80.894  41.269        0
    1  120.689 -41.269        1
    2   10.689 -41.269        3

Here, the column ``N_match`` indicates the number of matches found in catalog B for each object in catalog A.

To find the objects in catalog A that are also in catalog B, set the ``min_match`` parameter to 1:

.. code-block:: python

    print(result_object.get_dataframe1(min_match=1))

Expected output::

            Ra     Dec  N_match
    1  120.689 -41.269        1
    2   10.689 -41.269        3

The method :func:`pycorrelator.XMatchResult.get_dataframe1` returns a pandas DataFrame object.
So if you want to find the objects in catalog A that are not in catalog B, you can do the following
`pandas DataFrame operation <https://pandas.pydata.org/docs/user_guide/10min.html#boolean-indexing>`_:

.. code-block:: python

    df1 = result_object.get_dataframe1()
    print(df1[df1['N_match'] == 0])

Expected output::

            Ra     Dec  N_match
    0   80.894  41.269        0

get_dataframe2()
----------------

Similarly, to get the matching results of catalog B, use the :func:`pycorrelator.XMatchResult.get_dataframe2` method.
The usage is the same as :func:`pycorrelator.XMatchResult.get_dataframe1`. Just instead of giving the matching results
of each object in catalog A, it gives the matching results of each object in catalog B.

.. code-block:: python

    print(result_object.get_dataframe2())

Expected output::

            Ra     Dec  N_match
    0   10.688 -41.270        1
    1   10.689 -41.270        1
    2   10.690 -41.269        1
    3  120.690 -41.270        1

get_serial_dataframe()
----------------------

If you want to get the matching results of both catalogs in a single DataFrame, you can use the
:func:`pycorrelator.XMatchResult.get_serial_dataframe` method. For example:

.. code-block:: python

    print(result_object.get_serial_dataframe(min_match=0))

Expected output::

            Ra     Dec  N_match  is_cat1
    0   80.894  41.269        0     True
    1  120.689 -41.269        1     True
    3  120.690 -41.270       -1    False
    2   10.689 -41.269        3     True
    0   10.688 -41.270       -1    False
    1   10.689 -41.270       -1    False
    2   10.690 -41.269       -1    False

Here, the column ``is_cat1`` indicates whether the object is from catalog A (True) or catalog B (False).
And the column ``N_match`` indicates the number of matches found in catalog B for each object in catalog A.
Each object in catalog A is shown in order as in the input catalog, followed by the matching results of the objects in catalog B.
This means that if an object in catalog B is matches with multiple objects in catalog A, it will be shown multiple times.
And if an object in catalog B is not matched with any object in catalog A, it will not be shown in the output.

.. note::
    The ``N_match`` value is -1 for all objects in catalog B. This is designed for efficiency reasons.

Furthermore, if you want to make catalog B as the 'primary' catalog, you can set the ``reverse`` parameter to ``True``:

.. code-block:: python

    print(result_object.get_serial_dataframe(min_match=0, reverse=True))

Expected output::

            Ra     Dec  N_match  is_cat1
    0   10.688 -41.270        1    False
    2   10.689 -41.269       -1     True
    1   10.689 -41.270        1    False
    2   10.689 -41.269       -1     True
    2   10.690 -41.269        1    False
    2   10.689 -41.269       -1     True
    3  120.690 -41.270        1    False
    1  120.689 -41.269       -1     True

Here we can see that the third object (index of 2) in catalog A shown 3 times in the output,
because it has 3 matches in catalog B. And the first object (index of 0) in catalog A is not
shown in the output, because it has no matches in catalog B.

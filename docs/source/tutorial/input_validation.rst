What kind of input format does pycorrelator support?
====================================================


Currently, all of the :doc:`functions <../ref/index>` in pycorrelator support the following input format only:

   - pandas.DataFrame
   - numpy.ndarray


pandas.DataFrame
----------------

The input DataFrame should have the following columns:

   - One of the ``['ra', 'Ra', 'RA']`` (Right Ascension) in degrees.
   - One of the ``['dec', 'Dec', 'DEC']`` (Declination) in degrees.

Addtionally, the DataFrame can have any other columns as well. These columns will be preserved in the output.
And the index of the DataFrame has no restrictions and will be preserved in the output as well. (MultiIndex is not supported for now.)

numpy.ndarray
-------------

The input numpy array should be in the shape of (N, 2), where N is the number of objects in the catalog. The two columns should be:

   - The first column (``data[:, 0]``) should be the Right Ascension in degrees.
   - The second column (``data[:, 1]``) should be the Declination in degrees.

Addtional data columns are not supported in the numpy array format for now.

.. pycorrelator documentation master file, created by
   sphinx-quickstart on Sat Jun 15 21:23:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycorrelator's documentation!
========================================

**pycorrelator** is a Python package designed to perform cross correlation and self correlation analyses in spherical coordinates.
This is particularly useful in fields such as astrophysics, geophysics, and any domain where objects are naturally distributed on a spherical surface.
Currently, this package only support astrophysics coordinates (Ra, Dec) in degrees. More units and naming convention will be supported in the future.


Quickstart
----------

1. First, :doc:`install the package <install>`.
2. Second, make sure your data is in the :doc:`supported formats <tutorial/input_validation>`.
3. Finally,

   - If you want to perform a cross-matching between two catalogs, use the :ref:`xmatch <xmatch-ref>` function.
     See the :doc:`cross-matching example <tutorial/xmatch>` for how to use it.
   - If you want to cluster the objects in a catalog with the Friends-of-Friends (FoF) algorithm, use
     the :ref:`group_by_quadtree <fof-ref>` function. See the :doc:`clustering example <tutorial/fof>` for how to use it.
   - If you want to remove duplicates from a catalog, also using the :ref:`group_by_quadtree <fof-ref>` function.
     See the :doc:`duplicates removal example <tutorial/duplicates_removal>` for how to do it.


Contents
--------

.. toctree::
   :maxdepth: 2

   install
   tutorial/index
   ref/index
   dev/index

.. note::

   This project is under active development. If you find any issue, please report it at
   the GitHub repository's `issue tracker <https://github.com/technic960183/pycorrelator/issues>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

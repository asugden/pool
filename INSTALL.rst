************
Installation
************

.. Contents::

Introduction
============

Pool is a set of analysis functions and scripts built on top of Flow for the analysis
of data generated in the Andermann Lab.


Prerequisites
=============

* `Python <http://python.org>`_ 2.7
* `numpy <http://www.scipy.org>`_ >= 1.8
* `scipy <http://www.scipy.org>`_ >= 0.13.0
* `matplotlib <http://matplotlib.org>`_ >= 1.2.1
* `six <https://pypi.python.org/pypi/six>`_, initial attempts at Python 2/3 compatibility
* `pandas <http://pandas.pydata.org/s>`_
* `seaborn <https://pypi.python.org/pypi/seaborn>`_, pretty plotting

.. * `pycircstat <https://pypi.python.org/pypi/pycircstat>`_, deals with circular statistics.
.. * `shapely <https://pypi.python.org/pypi/Shapely>`_ >= 1.2.14 (**Windows users**: be sure to install from `Christophe Gohlke's built wheels <http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>`_)
.. * `OpenCV <http://opencv.org>`_ >= 2.4.8
.. * `h5py <http://www.h5py.org>`_ >= 2.2.1 (2.3.1 recommended), required for HDF5 file format

Depending on the features and data formats you wish to use, you may also need
to install the following packages:

* `bottleneck <http://pypi.python.org/pypi/Bottleneck>`_ >=0.8, for faster calculations
* `CouchDB <http://couchdb.apache.org/>`_, one option for caching analysis calculations

If you build the package from source, you may also need:

* `Cython <http://cython.org>`_
* `NLopt <https://nlopt.readthedocs.io/en/latest/>`_
* `OpenBLAS <http://www.openblas.net/>`_

If you want to generate the documentation, you will also need:

* `Sphinx <http://sphinx-doc.org>`_ >= 1.3.1

Installation
============

TODO

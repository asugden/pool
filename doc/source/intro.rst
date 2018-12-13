************
Introduction
************

.. Contents::

Pool contains all the lab-specific analysis code, particularly focused on
reactivation analysis. This library is build on top of, and heavily relies on, flow.


Using Pool
==========

TODO

Pool Structure
==============

**analyses**
	All possible analyses that can be run, with results being cached in a database.

**backends**
	Multiple possible analysis database backends. CouchDB is currently recommended,
	but requires separately setting up the database.

**layouts**
	Takes Mouse/Date/Run-Sorters and lays out plots on a page. Returns a figure.

**plotting**
	Puts data on an axes. First argument is usually a single axis.

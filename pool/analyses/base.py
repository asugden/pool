"""This abstract class lays out the structure of an analysis class."""

from six import with_metaclass
from abc import ABCMeta, abstractmethod

import flow.paths
from .. import config


class AnalysisBase(with_metaclass(ABCMeta, object)):
    """Base class for all analyses."""

    # Currently either 'classifier' or an empty string, depending on whether
    # or not the analysis requires classifier output
    requires = ['']

    # List of strings. All of the analyses that this particular analysis class
    # calculates.
    sets = []

    # List of other analyses that these analyses depend on. Currently not
    # implemented.
    depends_on = []

    # Level of analysis
    across = 'day'

    # Date of last modification. Used to determine if analyses needs to be
    # re-run (YYMMDD)
    updated = '000101'

    # Cache for lazy loading
    _mouse = ''
    _date = ''
    _t2ps = {}
    _c2ps = {}

    def __init__(self, data):
        """Default analysis init."""
        self._data = data

        # Check for lazy loading
        if self._data['mouse'] != self._mouse or self._data['date'] != self._date:
            self._mouse = self._data['mouse']
            self._date = self._data['date']
            self._t2ps = {}
            self._c2ps = {}

        self.out = self.run(self._data['mouse'], self._data['date'], self._data['training'],
                            self.data['running'], self._data['sated'], self._data['hungry'])

    @abstractmethod
    def run(self, mouse, date, training, running, sated, hungry):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        mouse : str
            mouse name
        date : str
            current date
        training : list of ints
            list of training run numbers as integers
        running : list of ints
            list of running-only run numbers as integers
        sated : list of ints
            list of sated spontaneous run numbers as integers
        hungry : list of ints
            list of hungry spontaneous run numbers as integers

        Returns
        -------
        dict
            All of the output values

        """
        return {}

    def trace2p(self, run):
        """Return a Trace2P instance for the shared day based on mouse and run."""

        if run not in self._t2ps:
            self._t2ps[run] = paths.gett2p(self._mouse, self._date, run)

        return self._t2ps[run]

    def classify2p(self, run):
        """Return a Classify2P instance for the shared day based on mouse and run."""

        if run not in self._c2ps:
            self._c2ps[run] = paths.classifier2p(self._mouse, self._date, run)

        return self._c2ps[run]

    def analysis(self, name):
        """Placeholder for calling the analysis database function."""
        return None

    def _get(self):
        """Return all analyses. Called by database.py"""

        return self.out

"""This abstract class lays out the structure of an analysis class."""

from six import with_metaclass
from abc import ABCMeta, abstractmethod


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

    def __init__(self, data):
        """Default analysis init."""
        self._data = data
        self.out = self._run_analyses()

    def get(self):
        """Return all analyses.

        :return: must return dict of outputs

        """
        return self.out

    @abstractmethod
    def _run_analyses(self):
        """Run all analyses and returns results in a dictionary."""
        return {}

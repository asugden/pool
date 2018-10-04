"""This abstract class lays out the structure of an analysis class."""

from six import with_metaclass
from abc import ABCMeta, abstractmethod
import numpy as np

from flow import paths

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

    # Level of analysis, can be 'day' or 'run'
    across = 'day'

    # Date of last modification. Used to determine if analyses needs to be
    # re-run (YYMMDD)
    updated = '000101'

    # Cache for lazy loading
    _mouse = None
    _date = None
    _run = None
    _mdobject = None

    def __init__(self, metadata_object, analysis_database, classifier_parameters):
        """Default analysis init."""
        self._andb = analysis_database
        self._pars = classifier_parameters

        # Check for lazy loading
        if (metadata_object.mouse != self._mouse
                or metadata_object.date != self._date
                or metadata_object.run != self._run):
            self._mouse = metadata_object.mouse
            self._date = metadata_object.date
            try:
                self._run = metadata_object.run
            except AttributeError:
                self._run = None
            self._mdobject = metadata_object

        self.out = self.run(self._mdobject)

    @abstractmethod
    def run(self, mdobject):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        mdobject : Date or Run instance
            mouse name

        Returns
        -------
        dict
            All of the output values

        """

        return {}

    def analysis(self, name):
        """
        Get an analysis from the analysis database.

        Parameters
        ----------
        name : str
            Analysis name

        Returns
        -------
        Output of analysis

        """
        return self._andb.get(name, self._mouse, self._date, self._run, force=True)

    def nanoutput(self):
        """
        Get a dictionary of all output values, pre-filled with NaNs

        Returns
        -------
        dict
            Dictionary of values from sets pre-filled with NaNs

        """

        return {key: np.nan for key in self._flatten(self.sets)}

    @property
    def pars(self):
        """
        Get parameters used for getting the current classifier2p.

        Returns
        -------
        dict
            Dictionary of all parameters
        """

        return self._pars

    def _get(self):
        """Return all analyses. Called by database.py"""

        return self.out

    def _flatten(self, nestedlist):
        """
        Convert a possibly nested list into a flat list

        :param nestedlist: possibly nested list
        :return: flattened list
        """

        if not isinstance(nestedlist, list):
            return [nestedlist]
        else:
            out = []
            for el in nestedlist:
                out.extend(self._flatten(el))
            return out

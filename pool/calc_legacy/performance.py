"""Analyses related to an animal's behavioral performance."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='value',
                format_string='hmm-dprime', format_args=[])
def dprime(date):
    """
    Return the inverse log p-value of drivenness of cells.

    Parameters
    ----------
    date : Date

    Result
    ------
    float

    """

    pass

@memoize_legacy(across='date', returns='value',
                format_string='hmm-criterion', format_args=[])
def criterion(date):
    """
    Return the inverse log p-value of drivenness of cells.

    Parameters
    ----------
    date : Date

    Result
    ------
    float

    """

    pass

@memoize_legacy(across='date', returns='value',
                format_string='hmm-engagement', format_args=[])
def engagement(date):
    """
    Return the inverse log p-value of drivenness of cells.

    Parameters
    ----------
    date : Date

    Result
    ------
    float

    """

    pass

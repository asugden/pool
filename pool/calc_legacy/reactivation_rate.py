"""Analyses of the frequency of reactivation."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='value',
                format_string='replay-freq-0.1-%s', format_args=['cs'])
def freq(date, cs):
    """
    Return the frequency of reactivation of a cs for a date.

    Parameters
    ----------
    date : Date
    cs : str

    Result
    ------
    float

    """

    pass


@memoize_legacy(across='date', returns='cell array',
                format_string='repcount-%.1f-%s', format_args=['threshold', 'cs'])
def cell(date, cs, threshold=0.1):
    """
    Return the reactivation count for a cell.

    Parameters
    ----------
    date : Date
    cs : str
    threshold :

    Result
    ------
    float

    """

    pass


@memoize_legacy(across='date', returns='cell matrix',
                format_string='reppair-%.1f-%s', format_args=['threshold', 'cs'])
def pair(date, cs, threshold=0.1):
    """
    Return the reactivation count for a cell.

    Parameters
    ----------
    date : Date
    cs : str
    threshold :

    Result
    ------
    float

    """

    pass

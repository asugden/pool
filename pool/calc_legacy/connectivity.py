"""Analyses of types of total connectivity."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='cell array',
                format_string='graph-clustering-%s', format_args=['cs'])
def total(date, cs):
    """
    Return the total connectivity of cells given a threshold of vdrive 50.

    Parameters
    ----------
    date : Date
    cs : str

    Result
    ------
    array

    """

    pass

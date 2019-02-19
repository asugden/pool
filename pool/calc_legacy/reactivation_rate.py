"""Analyses of the frequency of reactivation."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='value',
                format_string='replay-freq-0.1-%s', format_args=['cs'])
def freq(date, cs):
    """
    Return the inverse log p-value of drivenness of cells.

    Parameters
    ----------
    date : Date
    cs : str

    Result
    ------
    float

    """

    pass

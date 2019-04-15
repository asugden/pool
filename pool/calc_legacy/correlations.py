"""Analyses directly related to what cells are driven by."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='cell matrix',
                format_string='noise-correlation-%s', format_args=['cs'])
def noise(date, cs):
    """
    Return the inverse log p-value of drivenness of cells.

    Parameters
    ----------
    date : Date
    cs : str

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """

    pass

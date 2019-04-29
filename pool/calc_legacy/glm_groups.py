"""Analyses directly related to what cells are driven by."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='value',
                format_string='glmfrac-%s', format_args=['group'])
def fraction(date, group):
    """
    Return the inverse log p-value of drivenness of cells.

    Parameters
    ----------
    date : Date
    group : str {'ensure', 'ensure-vdrive', 'ensure-vdrive-nolick'}

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """

    pass

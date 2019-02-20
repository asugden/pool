"""Analyses directly related to what cells are driven by."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='cell array',
                format_string='cosdist-stim-ensure-%s-%i-%i-%s',
                format_args=['trace_type', 'start_s', 'end_s', 'cs'])
def stimulus(date, cs, trace_type='dff', start_s=0, end_s=2):
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

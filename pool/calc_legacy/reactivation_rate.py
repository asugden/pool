"""Analyses of the frequency of reactivation."""
from ..database_legacy import memoize_legacy


@memoize_legacy(across='date', returns='value',
                format_string='replay-freq%s-0.1-%s', format_args=['state', 'cs'])
def freq(date, cs, state=''):
    """
    Return the frequency of reactivation of a cs for a date.

    Parameters
    ----------
    date : Date
    cs : str
    state : '' for sated, '-hungry' for hungry

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


@memoize_legacy(across='date', returns='cell array',
                format_string='reward-specific-replay-%s', format_args=['cs'])
def cell_reward_rich(date, cs):
    """
    Return the reward-specific reactivation rate for each cell.

    Parameters
    ----------
    date
    cs

    Returns
    -------

    """

    pass


@memoize_legacy(across='date', returns='cell array',
                format_string='nonreward-specific-replay-%s', format_args=['cs'])
def cell_reward_poor(date, cs):
    """
    Return the reward-specific reactivation rate for each cell.

    Parameters
    ----------
    date
    cs

    Returns
    -------

    """

    pass


@memoize_legacy(across='run', returns='values',
                format_string='non-reward-specificity-plus-run%i', format_args=['run_number'])
def event_rich_poor(run, run_number):
    """
    Note that values >= 0.8 are rich and <= 0.2 are poor.

    Parameters
    ----------
    run

    Returns
    -------

    """

@memoize_legacy(across='run', returns='values',
                format_string='event-peaks-run%i-%.2f-%s', format_args=['run_number', 'threshold', 'cs'])
def events(run, run_number, cs, threshold=0.1):
    """
    Note that values >= 0.8 are rich and <= 0.2 are poor.

    Parameters
    ----------
    run

    Returns
    -------

    """

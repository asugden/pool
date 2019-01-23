"""General imaging functions."""
from ..database import memoize


@memoize(across='date', updated=190118)
def framerate(date):
    """
    Return the framerate of a Date object.

    Parameters
    ----------
    date : Date

    Returns
    -------
    float
        The framerate.

    """
    fr = 15.49
    for run in date.runs('training'):
        t2p = run.trace2p()
        fr = t2p.framerate
        break

    return fr

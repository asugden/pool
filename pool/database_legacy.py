from __future__ import division, print_function
from builtins import object

import functools
import inspect

from flow import paths
from replay.lib import analysis

_dbs = {}


def db(backend='couch', **kwargs):
    """Get a database of analyses from which one can pull individual analyses.

    :return: class that interacts with analysis db

    """

    global _dbs

    if backend not in _dbs:
        _dbs[backend] = analysis.db(backend)

    try:
        return _dbs[backend]
    except KeyError:
        raise ValueError("Unrecognized 'backend' option: {}".format(backend))


class memoize_legacy(object):
    """
    Memoization decorator. Takes in a translation string and argument order
    to create analysis names created by the replay library. Will return
    a warning if a recalculation is necessary.

    Parameters
    ----------
    across : {'date', 'run'}
        Defines if the analysis is a 'date' or 'run' analysis. Might be able to
        infer this from the parsed kwargs if all functions must adhere to a
        specific argument convention (must have either a 'date' or 'run'
        argument).
    updated : int
        Date of last update. Used to force a re-calculation if needed. If the
        stored date is different (doesn't check for older/newer), ignores
        stored value and recalculates.
    requires_classifier : bool
        If True, notes that the analysis requires the AODE classifier.
    returns : {'value', 'cell array', 'cell matrix', 'trial matrix'}, optional
        If the analysis returns either an array of cells or a matrix of cells
        and the trace2p is subset, then it will return the subset of the analysis
        results.

    Returns
    -------
    fn
        Returns a wrapped function that uses cached values if possible.

    Notes
    -----
        For analyses that require the classifier, this memoization gets the
        default parameters if they were not specifically passed in.

        The memoizer effectively adds a hidden 'force' argument to all
        memoized analyses. If True, cached values are ignored and the analysis
        is always recalculated.

    """

    def __init__(self, across, requires_classifier=False, returns='value',
                 format_string='', format_args=()):
        """Init."""
        self.across = across
        assert across in ['date', 'run']
        self.requires_classifier = requires_classifier
        self.returns = returns
        self.format_string = format_string
        self.format_args = format_args

        self.db = db()

    def __call__(self, fn):
        """Make the class behave like a function."""

        # Make the memoized function look like the original function upon
        # inspection.
        @functools.wraps(fn)
        def memoizer(*args, **kwargs):
            # Effectively adds a 'force' argument to all memoized functions.
            force = kwargs.pop('force', False)

            # Parse the args and kwargs into a single kwargs dict.
            parsed_kwargs = inspect.getcallargs(fn, *args, **kwargs)

            # Extract mouse/date/run
            if self.across == 'date':
                date_or_run = parsed_kwargs['date']
            elif self.across == 'run':
                date_or_run = parsed_kwargs['run']
                parsed_kwargs['run'] = date_or_run.run

            subset = date_or_run.cells

            format_tuple = []
            for key in self.format_args:
                format_tuple.append(parsed_kwargs[key])

            analysis_name = self.format_string % tuple(format_tuple)

            if subset is not None:
                date_or_run.set_subset(None)

            out = self.db.get(analysis_name, date_or_run.mouse, date_or_run.date, force=force)

            if subset is not None:
                date_or_run.set_subset(subset)

            if out is None:
                return None
            elif subset is not None and self.returns == 'cell array':
                return out[subset]
            elif subset is not None and self.returns == 'cell matrix':
                return out[subset, subset]
            else:
                return out

        return memoizer


def _test_db_read():
    """
    lprun -f pool.database._test_db_read -f pool.backends.couch_backend.Database.get -f pool.backends.couch_backend.Database._parse_doc pool.database._test_db_read()

    """
    import timeit

    couch = db('couch', database='analysis')
    shelve = db('shelve')
    memory = db('memory')
    cloudant = db('cloudant', database='analysis', user='admin', password='andermann')

    n = 100
    analysis = 'stim_dff_plus'  # ndarray
    analysis = 'sort_borders'  # dict
    mouse = 'OA178'
    date = 180702

    # Make sure it's stored in all
    print("Couch val: {}".format(couch.get(analysis, mouse, date)))
    print("Cloudant val: {}".format(cloudant.get(analysis, mouse, date)))
    print("Shelve val: {}".format(shelve.get(analysis, mouse, date)))
    print("Memory val: {}".format(memory.get(analysis, mouse, date)))

    get_couch = timeit.timeit(
        "d.get(analysis, mouse, date)",
        setup="from pool.database import db; d = db('couch', database='analysis'); " +
        "analysis, mouse, date ='{}', '{}', {}".format(analysis, mouse, date),
        number=n
    )
    get_cloudant = timeit.timeit(
        "d.get(analysis, mouse, date)",
        setup="from pool.database import db; d = db('cloudant', database='analysis'); " +
              "analysis, mouse, date ='{}', '{}', {}".format(analysis, mouse, date),
        number=n
    )
    get_shelve = timeit.timeit(
        "d.get(analysis, mouse, date)",
        setup="from pool.database import db; d = db('shelve'); " +
              "analysis, mouse, date ='{}', '{}', {}".format(analysis, mouse, date),
        number=n
    )
    get_memory = timeit.timeit(
        "d.get(analysis, mouse, date)",
        setup="from pool.database import db; d = db('memory'); " +
              "analysis, mouse, date ='{}', '{}', {}".format(analysis, mouse, date),
        number=n
    )

    print("Test command: db.get('{}', '{}', {})".format(analysis, mouse, date))
    print("get_couch: {} ({:.1f}x memory)".format(get_couch/n, get_couch/get_memory))
    print("get_cloudant: {} ({:.1f}x memory)".format(get_cloudant/n, get_cloudant/get_memory))
    print("get_shelve: {} ({:.1f}x memory)".format(get_shelve/n, get_shelve/get_memory))
    print("get_memory: {} ({:.1f}x memory)".format(get_memory/n, get_memory/get_memory))

if __name__ == '__main__':
    temp = db()
    print(temp.get('behavior_plus_orig', 'CB173', '160519'))
    # temp.save()
    # temp.close()

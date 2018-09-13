from .backends import MemoryBackend, ShelveBackend, CouchBackend
from .backends._cloudant_backend import CloudantBackend
from . import config

_dbs = {}


def db(backend=None, **kwargs):
    """Get a database of analyses from which one can pull individual analyses.

    :return: class that interacts with analysis db

    """
    global _dbs
    params = config.params()
    if backend is None:
        backend = params['backends']['backend']
    options = params['backends'].get('{}_options'.format(backend), dict())
    options.update(kwargs)

    if backend not in _dbs:
        if backend == 'shelve':
            _dbs['shelve'] = ShelveBackend(**options)
        elif backend == 'couch':
            _dbs['couch'] = CouchBackend(**options)
        elif backend == 'memory':
            _dbs['memory'] = MemoryBackend(**options)
        elif backend == 'cloudant':
            _dbs['cloudant'] = CloudantBackend(**options)

    try:
        return _dbs[backend]
    except KeyError:
        raise ValueError("Unrecognized 'backend' option: {}".format(backend))


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
    print(temp.get('rwa-plus', 'CB173', '160519'))
    # temp.save()
    # temp.close()

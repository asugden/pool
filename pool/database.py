from .backends import MemoryBackend, ShelveBackend, CouchBackend
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

    try:
        return _dbs[backend]
    except KeyError:
        raise ValueError("Unrecognized 'backend' option: {}".format(backend))

if __name__ == '__main__':
    temp = db()
    print(temp.get('rwa-plus', 'CB173', '160519'))
    # temp.save()
    # temp.close()

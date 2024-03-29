"""Backend using a CouchDB.

Requires:
- python couchdb library >= 1.2
- CouchDB >= 2.0

"""
from __future__ import division, print_function

from builtins import object, str, zip

import couchdb
from couchdb.mapping import DateTimeField
import datetime
from getpass import getuser
import numpy as np
import pandas as pd
from tempfile import TemporaryFile
from uuid import uuid4
# from json_tricks import numpy_encode, json_numpy_obj_hook

from .base_backend import BackendBase, keyname

try:
    from couchdb.util import StringIO
except ImportError:
    from io import BytesIO as StringIO

import sys
if sys.version_info.major == 3:
    unicode = str


class CouchBackend(BackendBase):
    """Analysis backend that stores data in a CouchDB database.

    For more information, see:
    - http://couchdb-python.readthedocs.io/en/latest/
    - http://couchdb.apache.org/

    """

    def _initialize(
            self, host='localhost', port=5984, database='analysis', user=None,
            password=None):
        self._database = Database(
            database, host=host, port=port, user=user, password=password)

    def __repr__(self):
        """Repr."""
        return "CouchBackend(host={}, port={}, database={})".format(
            self.database.host, self.database.port, self.database.name)

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store a value from running an analysis in the data store."""
        if depends_on is None:
            depends_on = {}

        _id = keyname(analysis_name, **keys)
        # print('Storing: {}'.format(_id))
        doc = dict(
            analysis=analysis_name,
            value=data,
            timestamp=timestamp(),
            user=getuser(),
            updated=int(updated),
            depends_on=depends_on,
            **keys)

        self.database.put(_id=_id, **doc)

    # implement db.update() eventually
    # def store_all(self, data_dict, keys):
    #     pass

    def recall(self, analysis_name, keys):
        """Return the value from the data store for a given analysis."""
        _id = keyname(analysis_name, **keys)
        dbentry = self.database.get(_id)
        if dbentry is None:
            return None, None, None

        out = dbentry.get('value', None)

        stored_updated = int(dbentry.get('updated'))
        depends_on = dbentry.get('depends_on', {})

        return out, stored_updated, depends_on

    #
    # Couch backend-specific functions
    #

    @property
    def database(self):
        """The underlying database."""
        return self._database

    def delete_mouse(self, mouse, no_action=True):
        """Delete all entries for a given mouse."""
        query = {
            "selector": {
                "mouse": mouse
            },
            "fields": ["_id"]
        }
        num = 0
        for row in self.database.find(query):
            if not no_action:
                self.database.delete(row['_id'])
            num += 1

        if num > 0:
            if no_action:
                print("Found documents for mouse {}: {}".format(mouse, num))
            else:
                print("Documents deleted for mouse {}: {}".format(mouse, num))
        else:
            print("No matching documents found for mouse {}.".format(
                mouse))

    def delete_analysis(self, analysis, no_action=True):
        """Delete all entries for a given analysis."""
        query = {
            "selector": {
                "analysis": analysis
            },
            "fields": ["_id"]
        }
        num = 0
        for row in self.database.find(query):
            if not no_action:
                self.database.delete(row['_id'])
            num += 1

        if num > 0:
            if no_action:
                print("Found documents for analysis {}: {}".format(
                    analysis, num))
            else:
                print("Documents deleted for analysis {}: {}".format(
                    analysis, num))
        else:
            print("No matching documents found for analysis {}.".format(
                analysis))


def timestamp():
    """Return the current time as a JSON-friendly timestamp."""
    return DateTimeField()._to_json(datetime.datetime.now())


def parse_timestamp(ts):
    """Convert a JSON-friendly timestamp back to a python datetime object."""
    return DateTimeField()._to_python(ts)


class Database(object):
    """Database class that handles connecting with the couchdb server."""

    def __init__(
            self, name, host='localhost', port=5984, user=None, password=None):
        """Initialize a new Database connection."""
        self.name = name
        self.host = host
        self.port = port

        self._server = couchdb.Server('http://{}:{}'.format(host, port))
        self.login(user, password)
        try:
            self._db = self._server[name]
        except couchdb.http.ResourceNotFound:
            # TODO: catch bad/missing authentication error
            self._server.create(name)
            self._db = self._server[name]
        # finally:
        #     # self.authenticate(user=user, password=password)
        #     self.login(user, password)

    def __contains__(self, _id):
        """Check if an _id already exists in database."""
        return _id in self._db

    def login(self, user=None, password=None):
        """Authenticate for restricted commands.

        couchdb.Server has a 'login' methods, but it doesn't seem to work
        all of the time. Explicitly storing the user/pass combo as the
        credentials works more consistently.

        """
        # self._user, self._password = (user, password)
        if user is not None and password is not None:
            self._server.resource.credentials = (user, password)
        # if user is not None and password is not None:
        #     result = self._server.login(user, password)
        # else:
        #     result = None
        # return result

    def put(self, _id=None, **data):
        """Store a value in the database."""
        new_data, attachment_data = _put_prep(data)
        _id, _rev = self._put_assume_new(unicode(_id), **new_data)
        if attachment_data is not None:
            doc = {'_id': _id, '_rev': _rev}
            temp_file = TemporaryFile()
            if new_data['value'] == '__attachment__':
                np.save(temp_file, attachment_data)
            else:  # new_data['value'] should be '__df_attachment__'
                attachment_data.to_pickle(temp_file, compression=None)
            temp_file.seek(0)
            self._db.put_attachment(
                doc, temp_file, filename='value',
                content_type='application/octet-stream')
        return _id, _rev

    # def put(self, _id=None, **data):
    #     """Store a value in the database."""
    #     new_data = self._put_prep(data)
    #     _id, _rev = self._put_assume_new(_id, **new_data)
    #     return _id, _rev

    # @staticmethod
    # def _put_prep(data):
    #     data['value'] = numpy_encode(data['value'])
    #     return data

    def _put_assume_new(self, _id=None, **data):
        """Store a value in the database.

        Attempts to immediately add doc and falls back to replace if needed.

        """
        if _id is None:
            _id = unicode(str(uuid4()))
        doc = dict(_id=_id, **data)
        #
        # for key in doc:
        #     if isinstance(doc[key], str):
        #         doc[key] = unicode(doc[key])
        try:
            _id, _rev = self._db.save(doc)
        except couchdb.http.ResourceConflict:
            # TODO: _rev is in header, don't need to get entire doc
            # Don't use self.get, don't want to actually download an attachment
            current_doc = self._db.get(unicode(_id))
            doc['_rev'] = current_doc.rev

            _id, _rev = self._db.save(doc)
        return _id, _rev

    # def _put_check_first(self, _id=None, **data):
    #     """Store a value in the database.

    #     Checks to see if _id already exists.

    #     """
    #     if _id is None:
    #         _id = str(uuid4())
    #     doc = dict(_id=_id, **data)
    #     if _id in self._db:
    #         current_doc = self.get(_id)
    #         doc['_rev'] = current_doc.rev
    #         return self._db.save(doc)
    #     else:
    #         return self._db.save(doc)

    # def _put_assume_replace(self, _id=None, **data):
    #     """Store a value in the database.

    #     Trys to get existing doc first, falls back to directly adding if it
    #     doesn't already exist.

    #     """
    #     if _id is None:
    #         _id = str(uuid4())
    #     doc = dict(_id=_id, **data)
    #     try:
    #         current_doc = self.get(_id)
    #     except couchdb.http.ResourceNotFound:
    #         _id, _rev = self._db.save(doc)
    #     else:
    #         # TODO: _rev is in header, don't need to get entire doc
    #         doc['_rev'] = current_doc.rev
    #         _id, _rev = self._db.save(doc)
    #     return _id, _rev

    # def _put_delete(self, _id=None, **data):
    #     """Store a value in the database.

    #     Trys to delete a pre-existing entry first and then add.

    #     """
    #     if _id is None:
    #         _id = str(uuid4())
    #     doc = dict(_id=_id, **data)
    #     try:
    #         del self._db[_id]
    #     except couchdb.http.ResourceNotFound:
    #         pass
    #     return self._db.save(doc)

    def get(self, _id):
        """Return the value from the data store for a given analysis."""
        doc = self._db.get(unicode(_id), default=None)
        return self._parse_doc(doc)

    def _parse_doc(self, doc):
        try:
            val = doc['value']
        except TypeError:
            assert doc is None
            return doc
        if val == '__attachment__':
            doc['value'] = np.load(self._get_attachment(doc['_id']))
        elif val == '__df_attachment__':
            doc['value'] = pd.read_pickle(self._get_attachment(doc['_id']))
        else:
            doc['value'] = _restore_nan(doc['value'])
        return doc

    def _get_attachment(self, _id):
        attachment = self._db.get_attachment(_id, 'value')
        if isinstance(attachment, couchdb.http.ResponseBody):
            attachment = StringIO(attachment.read())
        return attachment

    # @staticmethod
    # def _parse_doc(doc):
    #     doc['value'] = json_numpy_obj_hook(doc['value'])
    #     return doc

    def delete(self, _id):
        """Delete an entry by id."""
        del self._db[_id]

    def find(self, query):
        """Run a query on the database, returning a view."""
        return self._db.find(query)

    def compact(self):
        """Initiate compaction of current database.

        Deletes old document revisions and deleted documents.

        """
        result = self._db.compact()
        return result

    def pull(self, other):
        """Initiate replication that pulls changes from other database."""
        source = other._db.resource.url
        target = self._db.resource.url
        result = self._server.replicate(source, target)

        return result

    def push(self, other):
        """Initiate replication that pushed changes to other database."""
        source = self._db.resource.url
        target = other._db.resource.url
        result = self._server.replicate(source, target)

        return result

    def view(self, view):
        """Run and return result from a pre-defined view."""
        return self._db.view(view)


def _put_prep(data):
    """Check data before storing.

    Converts numpy number types to Python types.
    Pulls out numpy arrays and pandas DataFrames for binary storage.

    """
    val = data.get('value')
    if isinstance(val, np.ndarray):
        data['value'] = '__attachment__'
        return data, val
    elif isinstance(val, pd.DataFrame):
        data['value'] = '__df_attachment__'
        return data, val
    if isinstance(val, np.generic):
        # Should catch all numpy scalars and convert them to basic types
        val = val.item()
    data['value'] = _strip_nan(val)
    return data, None


def _strip_nan(val):
    """Replace NaN's.

    As a side-efect, converts all iterables to lists.

    """
    if isinstance(val, float) and np.isnan(val):
        return '__NaN__'
    elif isinstance(val, dict):
        return {key: _strip_nan(item) for key, item in val.items()}
    elif isinstance(val, list) or isinstance(val, tuple):
        return [_strip_nan(item) for item in val]
    elif isinstance(val, set):
        raise NotImplementedError
    return val


def _restore_nan(val):
    """Replace NaN's.

    As a side-effect, converts all iterables to lists.

    """
    if val == '__NaN__':
        return np.nan
    elif isinstance(val, dict):
        return {key: _restore_nan(item) for key, item in val.items()}
    elif isinstance(val, list) or isinstance(val, tuple):
        return [_restore_nan(item) for item in val]
    return val


def test_put():
    """Test time for various storing methods."""
    import timeit
    n = 100
    key = None  # {"'1'", 'None'}
    put1 = timeit.timeit(
        'db._put_check_first(key=key, **data)', setup="import replay.lib.couch; " +
        "data = {'int': 1, 'string': 'foo', 'list': [1, 2, 'bar']};" +
        "db = replay.lib.couch.Connection(host='tolman').database('testing');" +
        "key = {}".format(key), number=n)
    put2 = timeit.timeit(
        'db._put_assume_new(key=key, **data)', setup="import replay.lib.couch; " +
        "data = {'int': 1, 'string': 'foo', 'list': [1, 2, 'bar']};" +
        "db = replay.lib.couch.Connection(host='tolman').database('testing');" +
        "key = {}".format(key), number=n)
    put3 = timeit.timeit(
        'db._put_assume_replace(key=key, **data)', setup="import replay.lib.couch; " +
        "data = {'int': 1, 'string': 'foo', 'list': [1, 2, 'bar']};" +
        "db = replay.lib.couch.Connection(host='tolman').database('testing');" +
        "key = {}".format(key), number=n)
    put4 = timeit.timeit(
        'db._put_delete(key=key, **data)', setup="import replay.lib.couch; " +
        "data = {'int': 1, 'string': 'foo', 'list': [1, 2, 'bar']};" +
        "db = replay.lib.couch.Connection(host='tolman').database('testing');" +
        "key = {}".format(key), number=n)

    print("key = {}".format(key))
    print("put_check_first: {}".format(put1 / n))
    print("put_assume_new: {}".format(put2 / n))
    print("put_assume_replace: {}".format(put3 / n))
    print("put_delete: {}".format(put4 / n))


def test_put_numpy():
    """Compare time for storing and recalling numpy array and list."""
    import numpy as np
    import timeit

    import pool.backends.couch_backend as cb

    n = 100
    key = 'np_test'
    data_arr = {'value': [1, 2, 3, 4, 5]}
    data_np = {'value': np.arange(5)}

    db = cb.Database('testing', host='localhost')

    db.put(_id=key, **data_arr)
    returned = db.get(key)
    assert all(x == y for x, y in zip(returned['value'], data_arr['value']))

    db.put(_id=key, **data_np)
    returned = db.get(key)
    assert all(x == y for x, y in zip(returned['value'], data_np['value']))

    put_arr = timeit.timeit(
        'db.put(_id=key, **data)', setup="import pool.backends.couch_backend as cb; " +
        "data = {'value': [1, 2, 3]}; " +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)
    get_arr = timeit.timeit(
        'db.get(key)', setup="import pool.backends.couch_backend as cb; " +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)

    put_np = timeit.timeit(
        'db.put(_id=key, **data)', setup="import pool.backends.couch_backend as cb; import numpy; " +
        "data = {'value': numpy.arange(3)}; " +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)
    get_np = timeit.timeit(
        'db.get(key)', setup="import pool.backends.couch_backend as cb; " +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)

    put_np_as_arr = timeit.timeit(
        "data2 = {'value': data['value'].tolist()}; db.put(_id=key, **data2)", setup="import pool.backends.couch_backend as cb; import numpy; " +
        "data = {'value': numpy.arange(3)}; " +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)
    get_np_as_arr = timeit.timeit(
        'data = db.get(key); data = numpy.array(data)', setup="import pool.backends.couch_backend as cb; import numpy;" +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)

    print("key = {}".format(key))
    print("put_array: {}".format(put_arr / n))
    print("put_np: {}".format(put_np / n))
    print("put_np_as_arr: {}".format(put_np_as_arr / n))
    print("get_array: {}".format(get_arr / n))
    print("get_np: {}".format(get_np / n))
    print("get_np_as_arr: {}".format(get_np_as_arr / n))


def test_put_df():
    """Compare time for storing and recalling numpy array and list."""
    import numpy as np
    import pandas as pd
    import timeit

    import pool.backends.couch_backend as cb

    n = 100
    key = 'df_test'
    index = pd.MultiIndex.from_product([['foo', 'bar'], np.arange(3)])
    data = {'value': pd.DataFrame(
        {'a': np.random.random(6), 'b':np.random.random(6)}, index=index)}

    db = cb.Database('testing', host='localhost')

    db.put(_id=key, **data)
    returned = db.get(key)
    assert all(data['value'] == returned['value'])

    put_df = timeit.timeit(
        'db.put(_id=key, **data)', setup="import pool.backends.couch_backend as cb; import numpy, pandas;" +
        "index = pandas.MultiIndex.from_product([['foo', 'bar'], numpy.arange(3)]); " +
        "data = {'value': pandas.DataFrame({'a': numpy.random.random(6), 'b':numpy.random.random(6)}, index=index)};" +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)
    get_df = timeit.timeit(
        'db.get(key)', setup="import pool.backends.couch_backend as cb; " +
        "db = cb.Database('testing', host='localhost'); " +
        'key = "{}"'.format(key), number=n)

    print("key = {}".format(key))
    print("put_df: {}".format(put_df / n))
    print("get_df: {}".format(get_df / n))

if __name__ == '__main__':
    # test_put()
    # test_put_numpy()
    test_put_df()

    # db = CouchBackend(host='localhost')
    # print(db.get('qdist-run11-0.1-minus', 'OA32', 170417, force=True))
    # print(db.get('sort_borders', 'OA178', 180601, force=True))
    # print(db.get('dprime', 'AS20', 160816, force=True))

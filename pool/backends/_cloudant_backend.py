"""Initial attempts at replacing couchdb library with cloudant library.

Incomplete.

"""
from cloudant.client import CouchDB
from io import BytesIO
import numpy as np
from uuid import uuid4
# import urllib3
# import json
# import requests

from .base_backend import BackendBase, keyname


class CloudantBackend(BackendBase):
    """Analysis backend that stores data in a CouchDB database using cloudant.

    For more information, see:
    - *cloundant docs*
    - http://couchdb.apache.org/

    """

    def _initialize(
            self, host='localhost', port=5984, database='analysis', user=None,
            password=None):
        self._database = Database(
            user=user, password=password, name=database, host=host, port=port)

    def __repr__(self):
        return "CloudantBackend(host={}, port={}, database={})".format(
            self.database.host, self.database.port, self.database.name)

    def store(self, analysis_name, data, keys, dependents=None):
        pass

    def recall(self, analysis_name, keys):
        """Return the value from the data store for a given analysis."""

        dbentry = self.database.get(keyname(analysis_name, keys))
        if dbentry is None:
            return None, True

        out = dbentry.get('value', None)
        updated = dbentry.get('updated')
        return out, int(updated) != int(keys['updated'])

    def is_analysis_old(self, analysis_name, keys):
        """Determine if the analysis needs to be re-run."""
        key = keyname(analysis_name, keys)
        # TODO: check if old too
        return key not in self.database

    #
    # Cloudant backend-specific functions
    #

    @property
    def database(self):
        """The underlying database."""
        return self._database


def timestamp():
    """Return the current time as a JSON-friendly timestamp."""
    return DateTimeField()._to_json(datetime.datetime.now())


def parse_timestamp(ts):
    """Convert a JSON-friendly timestamp back to a python datetime object."""
    return DateTimeField()._to_python(ts)


class Database(object):
    """Database class that handles connecting with the couchdb server."""

    def __init__(
            self, user, password, name, host='localhost', port=5984):
        """Initialize a new Database connection."""
        self.name = name
        self.host = host
        self.port = port
        # self.doc_path = "http://{}:{}/{}".format(self.host, self.port, self.name)
        # self.http = urllib3.PoolManager()

        self._server = CouchDB(user, password, url='http://{}:{}'.format(
            self.host, self.port), connect=True, auto_renew=True)

        try:
            self._db = self._server[name]
        except KeyError:
            # TODO: catch bad/missing authentication error
            self._server.create_database(name)
            self._db = self._server[name]

    def login(self, user=None, password=None):
        """Authenticate for restricted commands."""
        self._server.session_login(user=user, passwd=password)

    def put(self, _id=None, **data):
        """Store a value in the database."""
        new_data, numpy_array = Database._put_prep(data)
        doc = self._put_assume_new(_id, **new_data)
        if numpy_array is not None:
            temp_file = TemporaryFile()
            np.save(temp_file, numpy_array)
            temp_file.seek(0)
            # TODO: check attachment success
            doc.put_attachment(
                attachment='value', content_type='application/octet-stream',
                data=temp_file)
        return doc

    @staticmethod
    def _put_prep(data):
        val = data.get('value')
        if isinstance(val, np.ndarray):
            data['value'] = '__attachment__'
            return data, val
        if isinstance(val, np.generic):
            # Should catch all numpy scalars and convert them to basic types
            val = val.item()
        data['value'] = Database._strip_nan(val)
        return data, None

    @staticmethod
    def _strip_nan(val):
        """Replace NaN's.

        As a side-efect, converts all iterables to lists.

        """
        if isinstance(val, float) and np.isnan(val):
            return '__NaN__'
        elif isinstance(val, dict):
            return {key: Database._strip_nan(item) for key, item in val.items()}
        elif isinstance(val, list) or isinstance(val, tuple):
            return [Database._strip_nan(item) for item in val]
        elif isinstance(val, set):
            raise NotImplementedError
        return val

    def _put_assume_new(self, _id=None, **data):
        """Store a value in the database.

        Attempts to immediately add doc and falls back to replace if needed.

        """
        if _id is None:
            _id = str(uuid4())
        doc = dict(_id=_id, **data)
        try:
            current_doc = self._db.create_document(doc, throw_on_exists=True)
        except couchdb.http.ResourceConflict:
            # TODO: _rev is in header, don't need to get entire doc
            # Don't use self.get, don't want to actually download an attachment
            current_doc = self._db.get(_id)
            current_doc.update(doc)
            current_doc.save()
        return current_doc

    def get(self, _id):
        """Return the value from the data store for a given analysis."""
        try:
            doc = self._db[_id]
            # For speed testing
            del self._db[_id]
        except KeyError:
            return None
        else:
            return self._parse_doc(doc)

    # def get(self, _id):
    #     """Return the value from the data store for a given analysis."""
    #     doc = requests.get(self.doc_path + '/{}'.format(_id)).json()
    #     return self._parse_doc(doc)

    # def get(self, _id):
    #     r = self.http.request('GET', self.doc_path + '/{}'.format(_id))
    #     doc = json.loads(r.data.decode('utf-8'))
    #     return self._parse_doc(doc)

    def _parse_doc(self, doc):
        if doc['value'] == '__attachment__':
            attachment = doc.get_attachment(attachment='value', attachment_type='binary')
            doc['value'] = np.load(BytesIO(attachment))
        else:
            doc['value'] = Database._restore_nan(doc['value'])
        return doc

    @staticmethod
    def _restore_nan(val):
        """Replace NaN's.

        As a side-effect, converts all iterables to lists.

        """
        if val == '__NaN__':
            return np.nan
        elif isinstance(val, dict):
            return {key: Database._restore_nan(item) for key, item in val.iteritems()}
        elif isinstance(val, list) or isinstance(val, tuple):
            return [Database._restore_nan(item) for item in val]
        return val


    def delete(self, _id):
        """Delete an entry by id."""
        self._db[_id].delete()


def test():
    cb = CloudantBackend(
        host='localhost', database='analysis', user='admin', password='andermann')
    from pudb import set_trace; set_trace()

if __name__ == '__main__':
    test()
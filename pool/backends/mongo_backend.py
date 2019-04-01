"""Backend using MongoDb."""
from __future__ import division, print_function

from builtins import object

from bson.binary import Binary
try:
    import cPickle as pickle
except ImportError:
    import pickle
from datetime import datetime
from getpass import getuser
import numpy as np
import pandas as pd
import pymongo

from .base_backend import BackendBase, keyname


# https://stackoverflow.com/questions/6367589/saving-numpy-array-in-mongodb

class MongoBackend(BackendBase):
    """Analysis backend that stores data in a MongoDB."""

    def _initialize(
            self, host='localhost', port=27017, database='analysis',
            collection='cache'):
        self._db = Connection(
            database, collection, host=host, port=port)

    @property
    def db(self):
        """The underlying Mongo database."""
        return self._db

    def __repr__(self):
        """Repr."""
        return ("MongoBackend(host={}, port={}, database={}, collection={})"
                .format(self.db.host, self.db.port, self.db.database_name,
                        self.db.collection_name))

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store a value from running an analysis in the data store."""
        if depends_on is None:
            depends_on = {}

        key = keyname(analysis_name, **keys)
        # print('Storing: {}'.format(key))
        doc = dict(
            _id=key,
            analysis=analysis_name,
            value=data,
            timestamp=datetime.now(),
            user=getuser(),
            updated=int(updated),
            depends_on=depends_on,
            **keys)

        self.db.put(doc=doc)

    def recall(self, analysis_name, keys):
        """Return the value from the data store for a given analysis."""
        key = keyname(analysis_name, **keys)

        dbentry = self.db.get(key)
        if dbentry is None:
            return None, None, None

        out = dbentry.get('value', None)

        stored_updated = int(dbentry.get('updated'))
        depends_on = dbentry.get('depends_on', {})

        return out, stored_updated, depends_on


class Connection(object):
    """Database class that handles connecting with the mongodb server."""

    def __init__(
            self, database, collection, host='localhost', port=27017):
        """Initialize a new Database connection."""
        self.database_name = database
        self.collection_name = collection
        self.host = host
        self.port = port

        self._client = pymongo.MongoClient(self.host, self.port, connect=True)

        # Initialize collection and indices if this is the first time
        if self.database_name not in self._client.list_database_names() or \
                self.collection_name not in self._client[
                self.database_name].list_collection_names():
            self.initialize_database()

        self._db = self._client[self.database_name]
        self.collection = self._db[self.collection_name]

    def initialize_database(self):
        """Create the database, collection, and indices."""
        db = self._client.get_database(self.database_name)
        collection = db.create_collection(self.collection_name)

    def put(self, doc):
        """Store a value in the database."""
        if isinstance(doc['value'], np.ndarray) or \
                isinstance(doc['value'], pd.DataFrame) or \
                isinstance(doc['value'], np.bool_):
            doc['__data__'] = Binary(pickle.dumps(doc['value'], protocol=2))
            doc['value'] = '__data__'
        self.collection.find_one_and_replace(
            {'_id': doc['_id']}, doc, upsert=True)

    def get(self, key):
        """Return the value from the data store for a given analysis."""
        doc = self.collection.find_one({'_id': key})
        return self._parse_doc(doc)

    def _parse_doc(self, doc):
        try:
            val = doc['value']
        except TypeError:
            assert doc is None
            return doc
        if val == '__data__':
            doc['value'] = pickle.loads(doc.pop('__data__'))
        return doc

    # def delete(self, _id):
    #     """Delete an entry by id."""
    #     del self._db[_id]

    # def find(self, query):
    #     """Run a query on the database, returning a view."""
    #     return self._db.find(query)

if __name__ == '__main__':

    db = MongoBackend(
        host='localhost', database='testing', collection='analysis')
    from pudb import set_trace; set_trace()

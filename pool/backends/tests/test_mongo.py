from numpy.testing import run_module_suite

from pool.backends.mongo_backend import MongoBackend
from pool.backends.tests.base_test import BaseTests

host = 'localhost'
database = 'testing'
collection = 'analysis'


class TestMongoDB(BaseTests):

    def setup(self):
        self.db = MongoBackend(
            host=host, database=database, collection=collection)
        self.keys = {'date': 180101,
                     'mouse': 'TM001'}
        self.updated = 180413

    def teardown(self):
        del self.db


if __name__ == "__main__":
    run_module_suite()

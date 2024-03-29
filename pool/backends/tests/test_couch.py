from numpy.testing import run_module_suite

from pool.backends.couch_backend import CouchBackend
from pool.backends.tests.base_test import BaseTests

host = 'localhost'
database = 'testing'


class TestCouchDB(BaseTests):

    def setup(self):
        self.db = CouchBackend(host=host, database=database)
        self.keys = {'date': 180101,
                     'mouse': 'TM001'}
        self.updated = 180413

    def teardown(self):
        del self.db


if __name__ == "__main__":
    run_module_suite()

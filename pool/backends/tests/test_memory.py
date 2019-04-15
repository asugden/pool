from numpy.testing import run_module_suite

from pool.backends.memory_backend import MemoryBackend
from pool.backends.tests.base_test import BaseTests


class TestCouchDB(BaseTests):

    def setup(self):
        self.db = MemoryBackend()
        self.keys = {'date': 180101,
                     'mouse': 'TM001'}
        self.updated = 180413

    def teardown(self):
        del self.db


if __name__ == "__main__":
    run_module_suite()

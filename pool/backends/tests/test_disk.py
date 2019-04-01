from numpy.testing import run_module_suite
from shutil import rmtree
from tempfile import mkdtemp

from pool.backends.disk_backend import DiskBackend
from pool.backends.tests.base_test import BaseTests


class TestCouchDB(BaseTests):

    def setup(self):
        self.savedir = mkdtemp()
        self.db = DiskBackend(savedir=self.savedir)
        self.keys = {'date': 180101,
                     'mouse': 'TM001'}
        self.updated = 180413

    def teardown(self):
        del self.db
        rmtree(self.savedir)


if __name__ == "__main__":
    run_module_suite()

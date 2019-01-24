from __future__ import print_function
import shelve
from copy import deepcopy
from time import sleep

from flow import paths

from .base_backend import BackendBase, keyname


class ShelveBackend(BackendBase):
    def _initialize(self):
        self.updated_analyses = {}
        self.dbs = {}
        self.dbrs = {}
        self.dbus = {}

    def store(self, analysis_name, data, keys, updated, dependents=None):
        """Store a value from running an analysis in the data store."""
        mouse = keys['mouse']
        self._open(mouse)
        key = keyname(analysis_name, **keys)
        self.dbrs[mouse][key] = data
        self.dbus[mouse][key] = deepcopy(updated)
        self.updated_analyses[mouse].append(key)

        for dependent in dependents:
            dependent_id = keyname(dependent, **keys)
            try:
                del self.dbrs[mouse][dependent_id]
                del self.dbus[mouse][dependent_id]
            except KeyError:
                pass
            else:
                self.updated_analyses[mouse].append(dependent_id)

    def recall(self, analysis_name, keys, updated):
        """Return the value from the data store for a given analysis."""
        mouse = keys.get('mouse')
        key = keyname(analysis_name, **keys)
        self._open(mouse)
        try:
            out = deepcopy(self.dbrs[mouse][key])
        except KeyError:
            return None, True
        except:
            sleep(10)
            out = deepcopy(self.dbrs[mouse][key])

        stored_updated = self.dbus.get(mouse, {}).get(key, 0)
        return out, int(updated) != int(stored_updated)

    def is_analysis_old(self, analysis_name, keys, updated):
        """Determine if the analysis needs to be re-run.

        Checks to see if analysis is already stored in shelve and the
        update key matches.

        """
        mouse = keys['mouse']
        key = keyname(analysis_name, **keys)
        self._open(mouse)
        return \
            key not in self.dbrs[mouse] or \
            key not in self.dbus[mouse] or \
            self.dbus[mouse][key] != updated

    def save(self, closedb=True):
        """Save all updated databases."""
        for mouse in self.dbs:
            if len(self.updated_analyses[mouse]) > 0:
                self.dbrs[mouse].sync()
                self.dbus[mouse].sync()
        if closedb:
            self._close()

    def _open(self, mouse):
        """Open the database."""
        if mouse not in self.dbrs:
            self.dbrs[mouse] = shelve.open(paths.db(mouse))
            self.dbus[mouse] = shelve.open(paths.udb(mouse))
            self.dbs[mouse] = {}
            self.updated_analyses[mouse] = []

    def _close(self):
        for mouse in self.dbrs:
            self.dbrs[mouse].close()
            self.dbus[mouse].close()

if __name__ == '__main__':
    db = ShelveBackend()
    print(db.get('dprime', 'AS20', 160816, force=False))

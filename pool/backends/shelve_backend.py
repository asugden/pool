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

    def store(self, analysis_name, data, keys, dependents=None):
        """Store a value from running an analysis in the data store."""
        mouse = keys['mouse']
        andate = deepcopy(keys['updated'])
        self._open(mouse)
        key = keyname(analysis_name, keys)
        self.dbrs[mouse][key] = data
        self.dbus[mouse][key] = andate
        self.updated_analyses[mouse].append(key)

        for dependent in dependents:
            dependent_id = keyname(dependent, keys)
            try:
                del self.dbrs[mouse][dependent_id]
                del self.dbus[mouse][dependent_id]
            except KeyError:
                pass
            else:
                self.updated_analyses[mouse].append(dependent_id)

    def recall(self, analysis_name, keys):
        """Return the value from the data store for a given analysis."""
        andate = keys.get('updated', -1)
        mouse = keys.get('mouse')
        key = keyname(analysis_name, keys)
        self._open(mouse)
        try:
            out = deepcopy(self.dbrs[mouse][key])
        except KeyError:
            return None, True
        except:
            sleep(10)
            out = deepcopy(self.dbrs[mouse][key])

        updated = self.dbus.get(mouse, {}).get(key, 0)
        return out, int(andate) != int(updated) and andate > 0

    def is_analysis_old(self, analysis_name, keys):
        """Determine if the analysis needs to be re-run.

        Checks to see if analysis is already stored in shelve and the
        update key matches.

        """
        andate = keys['updated']
        mouse = keys['mouse']
        key = keyname(analysis_name, keys)
        self._open(mouse)
        return \
            key not in self.dbrs[mouse] or \
            (len(andate) > 0 and key not in self.dbus[mouse]) or \
            (len(andate) > 0 and self.dbus[mouse][key] != andate)

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

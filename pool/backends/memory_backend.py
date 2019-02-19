from __future__ import print_function
from .base_backend import BackendBase, keyname


class MemoryBackend(BackendBase):

    def _initialize(self, **kwargs):
        """Initialization steps customizable by subclasses."""
        self._data = {}
        self._updated = {}
        self._depends_on = {}

    def __repr__(self):
        """Repr."""
        return "MemoryBackend()"

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store a value from running an analysis in the data store."""
        _id = keyname(analysis_name, **keys)
        self._data[_id] = data
        self._updated[_id] = updated
        self._depends_on[_id] = depends_on if depends_on is not None else {}

    def recall(self, analysis_name, keys):
        """Return the value from the data store for a given analysis."""
        _id = keyname(analysis_name, **keys)
        if _id in self._data:
            stored_updated = self._updated[_id]
            depends_on = self._depends_on[_id]
            return self._data[_id], stored_updated, depends_on
        return None, None, None


if __name__ == '__main__':
    db = MemoryBackend()
    print(db.get('hmm-LR2', 'AS20', 160816, force=True))

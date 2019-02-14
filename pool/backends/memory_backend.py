from __future__ import print_function
from .base_backend import BackendBase, keyname


class MemoryBackend(BackendBase):

    def _initialize(self, **kwargs):
        """Initialization steps customizable by subclasses."""
        self._data = {}
        self._depends_on = {}

    def __repr__(self):
        """Repr."""
        return "MemoryBackend()"

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store a value from running an analysis in the data store."""
        _id = keyname(analysis_name, **keys)
        self._data[_id] = data
        self._depends_on[_id] = depends_on if depends_on is not None else {}

    def recall(self, analysis_name, keys, updated):
        """Return the value from the data store for a given analysis."""
        _id = keyname(analysis_name, **keys)
        if _id in self._data:
            # This is a memory cache, so the stored update date will always
            # match the current update date.
            # Still needs to call needs_update to correctly log dependencies
            # for other calc functions.
            stored_updated = updated
            depends_on = self._depends_on[_id]
            return self._data[_id], self.needs_update(
                analysis_name, updated, stored_updated, depends_on)
        return None, True

    def is_analysis_old(self, analysis_name, keys, updated):
        """Determine if the analysis needs to be re-run."""
        return keyname(analysis_name, **keys) not in self._data


if __name__ == '__main__':
    db = MemoryBackend()
    print(db.get('hmm-LR2', 'AS20', 160816, force=True))

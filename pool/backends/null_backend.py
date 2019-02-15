from __future__ import print_function
from .base_backend import BackendBase


class NullBackend(BackendBase):

    def _initialize(self, **kwargs):
        """Initialization steps customizable by subclasses."""
        pass

    def __repr__(self):
        """Repr."""
        return "NullBackend()"

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store a value from running an analysis in the data store."""
        pass

    def recall(self, analysis_name, keys):
        """Return the value from the data store for a given analysis."""
        return None, None, None


if __name__ == '__main__':
    db = NullBackend()
    print(db.get('hmm-LR2', 'AS20', 160816, force=True))

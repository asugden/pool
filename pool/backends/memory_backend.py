from .base_backend import BackendBase, keyname


class MemoryBackend(BackendBase):

    def _initialize(self, **kwargs):
        """Initialization steps customizable by subclasses."""
        self._data = {}

    def store(self, analysis_name, data, keys, updated, dependents=None):
        """Store a value from running an analysis in the data store."""
        self._data[keyname(analysis_name, **keys)] = data

    def recall(self, analysis_name, keys, updated):
        """Return the value from the data store for a given analysis."""
        try:
            return self._data[keyname(analysis_name, **keys)], False
        except KeyError:
            return None, True

    def is_analysis_old(self, analysis_name, keys, updated):
        """Determine if the analysis needs to be re-run."""
        return keyname(analysis_name, **keys) not in self._data


if __name__ == '__main__':
    db = MemoryBackend()
    print(db.get('hmm-LR2', 'AS20', 160816, force=True))

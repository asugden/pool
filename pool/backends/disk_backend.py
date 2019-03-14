import cPickle as pickle
from getpass import getuser
import json
import numpy as np
import os.path as opath
import pandas as pd

import flow
from .base_backend import BackendBase, keyname
from .couch_backend import timestamp


class DiskBackend(BackendBase):
    """This backend writes data to the local filesystem."""

    def _initialize(self, savedir=None, **kwargs):
        if savedir is None:
            self.savedir = opath.join(flow.paths.outd, 'data_cache')
        else:
            self.savedir = savedir

    def __repr__(self):
        """Repr."""
        return "DiskBackend(savedir={})".format(self.savedir)

    def _filename(self, analysis_name, keys):
        """File location of desired analysis."""
        _id = keyname(analysis_name, **keys)
        file = opath.join(self.savedir, analysis_name, _id)
        flow.misc.mkdir_p(opath.dirname(file))
        return file

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store data."""
        if depends_on is None:
            depends_on = {}

        file = self._filename(analysis_name, keys)

        doc = dict(
            analysis=analysis_name,
            timestamp=timestamp(),
            user=getuser(),
            updated=int(updated),
            depends_on=depends_on,
            **keys)
        if isinstance(data, np.ndarray) or \
                isinstance(data, pd.DataFrame):
            doc['value'] = '__data__'
            with open(file + '.pkl', 'w') as f:
                pickle.dump(data, f, protocol=2)
        else:
            doc['value'] = data

        with open(file + '.json', 'w') as f:
            # Compact dump
            json.dump(doc, f, separators=(',', ':'))

    def recall(self, analysis_name, keys):
        """Recall data."""

        file = self._filename(analysis_name, keys)

        try:
            with open(file + '.json', 'r') as f:
                info = json.load(f)
        except IOError:
            return None, None, None

        if info['value'] == '__data__':
            with open(file + '.pkl', 'r') as f:
                data = pickle.load(f)
        else:
            data = info['value']

        stored_updated = info['updated']
        depends_on = info.get('depends_on', {})

        return data, stored_updated, depends_on

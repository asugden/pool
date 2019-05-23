try:
    import cPickle as pickle
except ImportError:
    import pickle
from getpass import getuser
import json
import os.path as opath

import flow
from .base_backend import BackendBase, keyname


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
        # Convert the id to a 16-character string
        _id = flow.misc.wordhash.hash(_id, use_new=True)[:16]
        file_base = opath.join(self.savedir, analysis_name)
        # If there's a mouse, date, and run in keys, group by mouse and date
        if 'mouse' in keys and 'date' in keys and 'run' in keys:
            file_base = opath.join(file_base, keys['mouse'], str(keys['date']))
        # If there's a mouse and date in keys (but no run), group by mouse
        elif 'mouse' in keys and 'date' in keys:
            file_base = opath.join(file_base, keys['mouse'])
        file = opath.join(file_base, _id)
        flow.misc.mkdir_p(opath.dirname(file))
        return file

    def store(self, analysis_name, data, keys, updated, depends_on=None):
        """Store data."""
        if depends_on is None:
            depends_on = {}

        file = self._filename(analysis_name, keys)

        doc = dict(
            analysis=analysis_name,
            timestamp=flow.misc.timestamp(),
            user=getuser(),
            updated=int(updated),
            depends_on=depends_on,
            value=data,
            **keys)
        try:
            with open(file + '.json', 'w') as f:
                # Compact dump
                json.dump(doc, f, separators=(',', ':'))
        except TypeError:
            doc['value'] = '__data__'
            with open(file + '.pkl', 'wb') as f:
                pickle.dump(data, f, protocol=2)
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
            with open(file + '.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            data = info['value']

        stored_updated = info['updated']
        depends_on = info.get('depends_on', {})

        return data, stored_updated, depends_on

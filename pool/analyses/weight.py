import numpy as np
import pandas as pd

from .base import AnalysisBase
from flow.misc import gsheets

class Weight(AnalysisBase):
    """
    Set the mouse's weight using information from a google doc
    """

    def _run_analyses(self):
        df = gsheets.dataframe('1rK6MAWhpgzPI4dRO7GtzzOvlmrrVcrlCSzL-IpquebA', self._data['mouse'], 'A1:J100')
        df[['date', 'weight']] = df[['date', 'weight']].apply(pd.to_numeric)

        out = {'weight': float(df.ix[df['date'] == self._data['date'], 'weight'])}
        wmn = np.nanmean(df['weight'])
        out['weight-scaled'] = (out['weight'] - wmn)/wmn

        return out

    sets = ['weight', 'weight-scaled']
    across = 'day'
    updated = '180411'

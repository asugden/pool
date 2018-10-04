from . import base


class Framerate(base.AnalysisBase):
    def run(self, date):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        date : Date object

        Returns
        -------
        dict
            All of the output values

        """

        out = {'framerate': 15.49}
        for run in date.runs('training'):
            t2p = run.trace2p()
            out['framerate'] = t2p.framerate
            break

        return out

    requires = ['']
    sets = ['framerate']
    across = 'day'
    updated = '181003'
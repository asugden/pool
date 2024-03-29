from __future__ import division
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np
import warnings

from . import base
from pool import config


class Behavior(base.AnalysisBase):
    def run(self, mouse, date, training, running, sated, hungry):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        mouse : str
            mouse name
        date : str
            current date
        training : list of ints
            list of training run numbers as integers
        running : list of ints
            list of running-only run numbers as integers
        sated : list of ints
            list of sated spontaneous run numbers as integers
        hungry : list of ints
            list of hungry spontaneous run numbers as integers

        Returns
        -------
        dict
            All of the output values

        """

        out = {}

        self.out = {}
        self.behavior(data)
        self.running(data)
        self.licking(data)
        self.solenoids(data)

        self.out['ensure-latency'] = self.eqlatency(data, 'ensure', None)
        self.out['quinine-latency'] = self.eqlatency(data, 'quinine', None)
        self.out['ensure-latency-all'] = self.eqlatency(data, 'ensure', 8)
        self.out['quinine-latency-all'] = self.eqlatency(data, 'quinine', 8)

        self.out['behavior-plus-pavlovian'] = self.behaviorpavlovian(data)

        return out

    requires = ['']
    sets = ['behavior_percent'] + \
           [['behavior_percent_%s',
             'behavior_presentations_%s',]
            for cs in pool.config.stimuli() + ['pavlovian']]
    across = 'day'
    updated = '181003'

    # ================================================================================== #
    # ANYTHING YOU NEED

    def behavior(self, data):
        vals = {}
        for r in data['train']:
            t2p = self.trace2p(r)

            for cs in ['plus', 'neutral', 'minus']:
                if cs not in vals:
                    vals[cs] = t2p.errors(cs)
                else:
                    vals[cs] = np.concatenate([vals[cs], t2p.errors(cs)])

        for cs in ['plus', 'neutral', 'minus']:
            self.out['behavior-%s' % (cs)] = np.nan if len(vals[cs]) == 0 else float(np.sum(vals[cs] == 0))/len(vals[cs])

        nhits = np.sum(vals['plus'] == 0)
        nplus = len(vals['plus'])
        nfas = np.sum(vals['neutral'] == 1) + np.sum(vals['minus'] == 1)
        npassives = len(vals['neutral']) + len(vals['minus'])

        dprime = norm.ppf((nhits + 0.5)/(nplus + 1.0)) - norm.ppf((nfas + 0.5)/(npassives + 1.0))
        self.out['behavior'] = float(nhits + (npassives - nfas))/(nplus + npassives)
        self.out['dprime'] = dprime
        self.out['dprimez'] = norm.cdf(dprime)

        z_hit_rate = norm.ppf((nhits + 0.5) / (nplus + 1.0))
        z_fa_rate = norm.ppf((np.sum(vals['minus'] == 1) + 0.5) / (len(vals['minus']) + 1.0))

        self.out['LR'] = np.exp((z_fa_rate**2 - z_hit_rate**2) / 2.)
        self.out['criterion'] = -1 * (z_hit_rate + z_fa_rate) / 2.

        for cs in ['plus', 'neutral', 'minus']:
            self.out['behavior-residual-%s'%(cs)] = self.out['behavior-%s'%(cs)] - self.out['behavior']

    def behaviorpavlovian(self, data):
        """
        Get the data for plus-pavlovian
        :param data:
        :return:
        """

        vals = []
        for r in data['train']:
            t2p = self.trace2p(r)
            vals = np.concatenate([vals, t2p.errors('plus'), [0]*t2p.ncs('pavlovian')])

        return np.nan if len(vals) == 0 else float(np.sum(vals == 0))/len(vals)

    def running(self, data):
        """
        Get the amount of total running and the running per CS
        :param data:
        :return:
        """

        run = 0
        csrun = {}
        for r in data['train']:
            t2p = self.trace2p(r)
            running = t2p.speed()
            run += np.mean(running)/len(data['train'])

            for cs in ['plus', 'neutral', 'minus']:
                if cs not in csrun: csrun[cs] = []

                ons = t2p.csonsets(cs)
                nframes = int(round(t2p.framerate*2.0))
                for on in ons:
                    if len(running) > on + nframes:
                        csrun[cs].append(running[on:on+nframes])
        self.out['running'] = run
        for cs in ['plus', 'neutral', 'minus']:
            if len(csrun[cs]) == 0: self.out['running-%s' % (cs)] = None
            else: self.out['running-%s' % (cs)] = np.mean(np.array(csrun[cs]).flatten())

    def licking(self, data):
        """
        Get the amount of total running and the running per CS
        :param data:
        :return:
        """

        runlick = 0
        latencies = []
        n = 0
        for r in data['train']:
            t2p = self.trace2p(r)
            licking = t2p.licking()
            mxlick = np.max(licking) if len(licking) > 0 else np.nan
            runlick += len(licking)

            ons = t2p.csonsets('plus', errortrials=0)
            nframes = int(round(t2p.framerate*8.0 - 0.5))
            for on in ons:
                if len(licking) > on + nframes:
                    if on > mxlick:
                        latencies.append(nframes)
                    else:
                        nlick = np.argmax(licking > on)
                        if licking[nlick] < on + nframes:
                            latencies.append(licking[nlick] - on)
                        else:
                            latencies.append(nframes)

        self.out['training-licks'] = runlick
        self.out['lick-latency'] = np.nan if len(latencies) == 0 else np.nanmedian(latencies)/t2p.framerate*1000

    def eqlatency(self, data, lattype='ensure', setzeros=None):
        """
        Get the amount of total running and the running per CS
        :param data:
        :return:
        """

        runlick = 0
        latencies = []
        n = 0
        for r in data['train']:
            t2p = self.trace2p(r)
            if lattype == 'ensure': beh = t2p.ensure()
            else: beh = t2p.quinine()

            ons = t2p.csonsets('plus' if lattype == 'ensure' else 'minus', errortrials=-1)
            nframes = int(round(t2p.framerate*8.0 - 0.5))
            for on in ons:
                beh = beh[beh >= on]

                if len(beh > 0):
                    if beh[0] - on < nframes:
                        latencies.append(float(beh[0] - on)/t2p.framerate)
                    elif setzeros is not None:
                        latencies.append(setzeros)

        return np.nanmean(latencies)

    def solenoids(self, data):
        """
        Set the number of ensure and quinine presentations given
        :param data:
        :return:
        """

        self.out['n-ensure'], self.out['n-quinine'] = 0, 0
        for r in data['train']:
            t2p = self.trace2p(r)
            self.out['n-ensure'] += np.sum(t2p.ensure() > 0)
            self.out['n-quinine'] += np.sum(t2p.quinine() > 0)
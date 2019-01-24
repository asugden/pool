from __future__ import division

import numpy as np
from scipy.stats import norm

import pool

from . import base

class Behavior(base.AnalysisBase):
    requires = ['']
    sets = ['behavior_%s_orig'%cs for cs in pool.config.stimuli()] + \
            ['behavior_orig', 'behavior_dprime_orig', 'behavior_dprimez_orig'] + \
            ['behavior_LR_orig', 'behavior_criterion_orig'] + \
            ['behavior_residual_%s_orig' % cs for cs in pool.config.stimuli()]
    across = 'day'
    updated = '181028'

    def run(self, date):
        out = {}
        out.update(self.behavior_original(date))

        return out

    def behavior_original(self, date):
        out, vals = {}, {}
        for run in date.runs(['training']):
            t2p = run.trace2p()

            for cs in ['plus', 'neutral', 'minus']:
                if cs not in vals:
                    vals[cs] = t2p.errors(cs)
                else:
                    vals[cs] = np.concatenate([vals[cs], t2p.errors(cs)])

        for cs in ['plus', 'neutral', 'minus']:
            out['behavior_%s_orig' % (cs)] = np.nan if len(vals[cs]) == 0 else float(np.sum(vals[cs] == 0)) / len(vals[cs])

        nhits = np.sum(vals['plus'] == 0)
        nplus = len(vals['plus'])
        nfas = np.sum(vals['neutral'] == 1) + np.sum(vals['minus'] == 1)
        npassives = len(vals['neutral']) + len(vals['minus'])

        dprime = norm.ppf((nhits + 0.5) / (nplus + 1.0)) - norm.ppf((nfas + 0.5) / (npassives + 1.0))
        out['behavior_orig'] = float(nhits + (npassives - nfas)) / (nplus + npassives)
        out['behavior_dprime_orig'] = dprime
        out['behavior_dprimez_orig'] = norm.cdf(dprime)

        z_hit_rate = norm.ppf((nhits + 0.5) / (nplus + 1.0))
        z_fa_rate = norm.ppf((np.sum(vals['minus'] == 1) + 0.5) / (len(vals['minus']) + 1.0))

        out['behavior_LR_orig'] = np.exp((z_fa_rate ** 2 - z_hit_rate ** 2) / 2.)
        out['behavior_criterion_orig'] = -1 * (z_hit_rate + z_fa_rate) / 2.

        for cs in ['plus', 'neutral', 'minus']:
            out['behavior_residual_%s_orig' % (cs)] = out['behavior_%s_orig' % (cs)] - out['behavior_orig']

        return out

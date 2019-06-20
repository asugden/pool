import numpy as np
from scipy.stats import norm

from flow import Run

from ..database import memoize

# @memoize(across='mouse', updated='')
def dprime(mouse, combine_passives=True, newpars=None):
    psy = mouse.psytracker(newpars=newpars)

    # Gather all the oris for this mouse and their index within the inputs
    oris = {key[4:] for key in psy.weight_labels if key[:4] == 'ori_'}
    input_idxs = {int(ori):
                  psy.weight_labels.index('ori_' + str(ori)) for ori in oris}

    # We are going to generate 3 input matrices for plus/neutral/minus
    # which we will use to predict lick rate at every trial, if that trial
    # had been plus/neutral/minus
    inputs = np.zeros_like(psy.inputs)
    if 'bias' in psy.weight_labels:
        inputs[:, psy.weight_labels.index('bias')] = 1
    plus_inputs = inputs.copy()
    neutral_inputs = inputs.copy()
    minus_inputs = inputs.copy()

    cum_trials = np.cumsum(psy.data['dayLength'])
    total_trials = inputs.shape[0]
    day_idx, run_idx = 0, 0
    start_trial = 0
    plus_ori, neutral_ori, minus_ori = -1, -1, -1
    while day_idx < len(psy.data['days']) and run_idx < len(psy.data['runs']):
        date = psy.data['days'][day_idx]
        run = psy.data['runs'][run_idx]
        run_length = psy.data['runLength'][run_idx]

        run_obj = Run(mouse.mouse, date, run)
        t2p = run_obj.trace2p()
        # If an ori wasn't shown this run, keep the previous association
        if 'plus' in t2p.d['orientations']:
            plus_ori = t2p.d['orientations']['plus']
        if 'neutral' in t2p.d['orientations']:
            neutral_ori = t2p.d['orientations']['neutral']
        if 'minus' in t2p.d['orientations']:
            minus_ori = t2p.d['orientations']['minus']
        # If the first run is missing one of these...choose better runs.
        assert(plus_ori >= 0)
        assert(neutral_ori >= 0)
        assert(minus_ori >= 0)

        plus_inputs[
            start_trial:start_trial + run_length, input_idxs[plus_ori]] = 1
        neutral_inputs[
            start_trial:start_trial + run_length, input_idxs[neutral_ori]] = 1
        minus_inputs[
            start_trial:start_trial + run_length, input_idxs[minus_ori]] = 1

        # Move indexes up a date/run as needed
        run_idx += 1
        start_trial += run_length
        if start_trial == cum_trials[day_idx]:
            # Next date
            day_idx += 1
        elif start_trial > total_trials or start_trial > cum_trials[day_idx]:
            raise ValueError("Malformed data, unable to match trials.")

    # The lick probability on each trial will be our predicted hit/FA rate
    hit_rate = psy.predict(data=plus_inputs)
    if combine_passives:
        fa_rate = (psy.predict(data=minus_inputs) +
                   psy.predict(data=neutral_inputs)) / 2.
    else:
        fa_rate = psy.predict(data=minus_inputs)

    # Calc dprime
    z_hit_rate = norm.ppf(hit_rate)
    z_fa_rate = norm.ppf(fa_rate)

    return z_hit_rate - z_fa_rate

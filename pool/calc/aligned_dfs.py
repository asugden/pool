"""Aligned DataFrames."""
import numpy as np
import pandas as pd

from .. import config
from ..database import memoize

POST_PAD_S = 2.3
POST_PAVLOVIAN_PAD_S = 2.6
PRE_PAD_S = 0.2


# Eventually would like to have a way to locally cache DataFrames to disk.
@memoize(across='run', updated=190213, large_output=True)
def trial_classifier_probability(run, pad_s=(PRE_PAD_S, POST_PAD_S)):
    """
    Return the classifier probability aligned to trial onsets.

    Parameters
    ----------
    run : Run
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, time
        Columns : {replay_type}

    """
    c2p = run.classify2p()
    t2p = run.trace2p()

    classifier_results = c2p.results()
    all_onsets = t2p.csonsets()
    replay_types = config.stimuli()

    prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)
    next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)

    fr = t2p.framerate

    pre_onset_pad_fr = int(np.ceil(pad_s[0] * fr))
    post_onset_pad_fr = int(np.ceil(pad_s[1] * fr))

    result = [pd.DataFrame()]
    for trial_idx, (onset, next_onset, prev_onset) in enumerate(
            zip(all_onsets, next_onsets, prev_onsets)):

        start_fr = prev_onset + post_onset_pad_fr
        end_fr = next_onset - pre_onset_pad_fr
        pre_fr = onset - start_fr

        trial_result = [pd.DataFrame()]
        for replay_type in replay_types:
            trial_replay_result = classifier_results[replay_type][
                start_fr:end_fr - 1]
            time = (np.arange(len(trial_replay_result)) - pre_fr) / fr

            index = pd.MultiIndex.from_product(
                [[run.mouse], [run.date], [run.run], [trial_idx], time],
                names=['mouse', 'date', 'run', 'trial_idx', 'time'])
            trial_result.append(
                pd.Series(trial_replay_result, index=index,
                          name=replay_type))
        result.append(pd.concat(trial_result, axis=1))
    final_result = pd.concat(result, axis=0)

    return final_result

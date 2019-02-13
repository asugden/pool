"""Trial DataFrames."""
from __future__ import division
from builtins import zip

import numpy as np
import pandas as pd

import pool

from . import base

# For trial-aligned DataFrames, ensures that the previous and next stim
# are not included.
POST_ONSET_PAD_S = 2.6  # 2.3 should work for all trials, except pavlovians
PRE_ONSET_PAD_S = 0.2

# For non-trial-aligned DataFrames, amount of time to include around each
# event.
PRE_S = 5.
POST_S = 5.


class TrialDf(base.AnalysisBase):
    """Trial-aligned reactivation DataFrames."""

    requires = ['classifier']
    sets = \
        ['trialdf_frames_inactmask',
         'trialdf_frames_noinactmask'] + \
        ['trialdf_events_0.1_xmask_inactmask',
         'trialdf_events_0.1_xmask_noinactmask',
         'trialdf_events_0.1_noxmask_inactmask',
         'trialdf_events_0.1_noxmask_noinactmask'] + \
        ['trialdf_events_0.05_xmask_inactmask',
         'trialdf_events_0.05_xmask_noinactmask',
         'trialdf_events_0.05_noxmask_inactmask',
         'trialdf_events_0.05_noxmask_noinactmask']

    across = 'run'
    updated = '190208'

    def run(self, run):
        """Run everything."""
        out = {}

        # Events relative to stimulus time
        for xmask in [True, False]:
            for inactivity_mask in [True, False]:
                for threshold in [0.1, 0.05]:
                    out.update(self.events(
                        run, threshold=threshold, xmask=xmask,
                        inactivity_mask=inactivity_mask))

        # Frames imaged, relative to stimulus time
        for inactivity_mask in [True, False]:
            out.update(self.frames(run, inactivity_mask=inactivity_mask))

        return out

    def events(self, run, threshold, xmask, inactivity_mask):
        """
        Determine event times relative to stimulus onset.

        Parameters
        ----------
        run : Run
        threshold : float
            Classifier cutoff probability.
        xmask : bool
            If True, only allow one event (across types) per time bin.
        inactivity_mask : bool
            If True, enforce that all events are during times of inactivity.

        Note
        ----
        Individual events will appear in this DataFrame multiple times!
        Events will show up both as being after a stimulus and before
        the next one.

        """
        out = {}

        t2p = run.trace2p()

        all_onsets = t2p.csonsets()

        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

        fr = t2p.framerate
        next_onset_pad_fr = int(np.ceil(PRE_ONSET_PAD_S * fr))
        prev_onset_pad_fr = int(np.ceil(POST_ONSET_PAD_S * fr))

        events = pool.dataframes.reactivation.events_df(
            [run], threshold, xmask=xmask,
            inactivity_mask=inactivity_mask)

        result = [pd.DataFrame()]
        for trial_idx, (onset, next_onset, prev_onset) in enumerate(zip(
                all_onsets, next_onsets, prev_onsets)):

            trial_events = events.loc[
                (events.frame >= (prev_onset + prev_onset_pad_fr)) &
                (events.frame < (next_onset - next_onset_pad_fr))].copy()
            trial_events['time'] = (trial_events.frame - onset) / fr
            trial_events['trial_idx'] = trial_idx

            result.append(trial_events)

        result_df = (pd
                     .concat(result, axis=0)
                     .rename(columns={'frame': 'abs_frame'})
                     .sort_index()
                     )

        analysis = 'trialdf_events_{}_{}_{}'.format(
            threshold,
            'xmask' if xmask else 'noxmask',
            'inactmask' if inactivity_mask else 'noinactmask')

        out[analysis] = result_df

        return out

    def frames(self, run, inactivity_mask):
        """
        Determine frame times relative to stimulus onset.

        Parameters
        ----------
        run : Run
        inactivity_mask : bool
            If True, enforce that all events are during times of inactivity.

        """
        out = {}

        t2p = run.trace2p()

        all_onsets = t2p.csonsets()

        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

        fr = t2p.framerate
        next_onset_pad_fr = int(np.ceil(PRE_ONSET_PAD_S * fr))
        prev_onset_pad_fr = int(np.ceil(POST_ONSET_PAD_S * fr))

        frames = (pool.dataframes.imaging
                  .frames_df([run], inactivity_mask)
                  .reset_index(['frame'])
                  )

        result = [pd.DataFrame()]
        for trial_idx, (onset, next_onset, prev_onset) in \
                enumerate(zip(all_onsets, next_onsets, prev_onsets)):
            # Pull out events around the current onset
            trial_frames = frames.loc[
                (frames.frame >= (prev_onset + prev_onset_pad_fr)) &
                (frames.frame < (next_onset - next_onset_pad_fr))].copy()

            # Convert to relative times
            trial_frames.frame -= onset

            # Add in some additional information
            trial_frames['time'] = \
                trial_frames.frame * trial_frames.frame_period
            trial_frames['trial_idx'] = trial_idx

            result.append(trial_frames)

        result_df = (pd
                     .concat(result, axis=0)
                     .set_index(['trial_idx', 'frame'], append=True)
                     )

        analysis = 'trialdf_frames_{}'.format(
            'inactmask' if inactivity_mask else 'noinactmask')

        out[analysis] = result_df

        return out

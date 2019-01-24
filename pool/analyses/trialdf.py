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
    sets = ['trialdf_classifier'] + \
        ['trialdf_frames_inactmask',
         'trialdf_frames_noinactmask'] + \
        ['trialdf_events_0.1_xmask_inactmask',
         'trialdf_events_0.1_xmask_noinactmask',
         'trialdf_events_0.1_noxmask_inactmask',
         'trialdf_events_0.1_noxmask_noinactmask'] + \
        ['trialdf_events_0.05_xmask_inactmask',
         'trialdf_events_0.05_xmask_noinactmask',
         'trialdf_events_0.05_noxmask_inactmask',
         'trialdf_events_0.05_noxmask_noinactmask'] + \
        ['trialdf_reward_events_0.1_xmask_inactmask',
         'trialdf_reward_events_0.1_xmask_noinactmask',
         'trialdf_reward_events_0.1_noxmask_inactmask',
         'trialdf_reward_events_0.1_noxmask_noinactmask'] + \
        ['trialdf_punishment_events_0.1_xmask_inactmask',
         'trialdf_punishment_events_0.1_xmask_noinactmask',
         'trialdf_punishment_events_0.1_noxmask_inactmask',
         'trialdf_punishment_events_0.1_noxmask_noinactmask'] + \
        ['trialdf_lickbout_events_0.1_xmask_inactmask',
         'trialdf_lickbout_events_0.1_xmask_noinactmask',
         'trialdf_lickbout_events_0.1_noxmask_inactmask',
         'trialdf_lickbout_events_0.1_noxmask_noinactmask'] + \
        ['trialdf_reward_events_0.05_xmask_inactmask',
         'trialdf_reward_events_0.05_xmask_noinactmask',
         'trialdf_reward_events_0.05_noxmask_inactmask',
         'trialdf_reward_events_0.05_noxmask_noinactmask'] + \
        ['trialdf_punishment_events_0.05_xmask_inactmask',
         'trialdf_punishment_events_0.05_xmask_noinactmask',
         'trialdf_punishment_events_0.05_noxmask_inactmask',
         'trialdf_punishment_events_0.05_noxmask_noinactmask'] + \
        ['trialdf_lickbout_events_0.05_xmask_inactmask',
         'trialdf_lickbout_events_0.05_xmask_noinactmask',
         'trialdf_lickbout_events_0.05_noxmask_inactmask',
         'trialdf_lickbout_events_0.05_noxmask_noinactmask']

    across = 'run'
    updated = '190111'

    def run(self, run):
        """Run everything."""
        out = {}

        # Classifier probability trace across trials.
        out.update(self.classifier(run))

        # Events relative to stimulus time
        for xmask in [True, False]:
            for inactivity_mask in [True, False]:
                for threshold in [0.1, 0.05]:
                    out.update(self.events(
                        run, threshold=threshold, xmask=xmask,
                        inactivity_mask=inactivity_mask))
                    for trigger in ['reward', 'punishment', 'lickbout']:
                        out.update(self.aligned_events(
                            run, trigger, threshold=threshold, xmask=xmask,
                            inactivity_mask=inactivity_mask))

        # Frames imaged, relative to stimulus time
        for inactivity_mask in [True, False]:
            out.update(self.frames(run, inactivity_mask=inactivity_mask))

        return out

    def classifier(self, run):
        """
        Calculate the classifier probabilities around each stimulus.

        Should probably add arguments for prev/post padding.

        """
        c2p = run.classify2p()
        t2p = run.trace2p()

        classifier_results = c2p.results()
        all_onsets = t2p.csonsets()
        # conditions = t2p.conditions()
        replay_types = pool.config.stimuli()

        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)
        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)

        fr = t2p.framerate

        next_onset_pad_fr = int(np.ceil(PRE_ONSET_PAD_S * fr))
        prev_onset_pad_fr = int(np.ceil(POST_ONSET_PAD_S * fr))

        result = [pd.DataFrame()]
        for trial_idx, (onset, next_onset, prev_onset) in enumerate(
                zip(all_onsets, next_onsets, prev_onsets)):

            start_fr = prev_onset + prev_onset_pad_fr
            end_fr = next_onset - next_onset_pad_fr
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

        # Merge in behavior results
        # behav_df = pool.dataframes.behavior.behavior_df([run])
        # final_result = (pool.dataframes.smart_merge(final_result, behav_df,
        #                                             how='left')
        #                 .set_index('error', append=True)
        #                 .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
        #                                  'condition', 'error', 'time'])
        #                 .sort_index()
        #                 )

        out = {'trialdf_classifier': final_result}
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
        # conditions = t2p.conditions()

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
            trial_events['frame'] -= onset
            trial_events['time'] = trial_events.frame / fr
            trial_events['trial_idx'] = trial_idx

            # add in trial_idx, condition
            # trial_events = pd.concat(
            #     [trial_events], keys=[trial_idx], names=['trial_idx'])
            # trial_events = pd.concat(
            #     [trial_events], keys=[cond], names=['condition'])

            result.append(trial_events)

        result_df = pd.concat(result, axis=0)
        # result_df = result_df.reorder_levels(
        #     ['mouse', 'date', 'run', 'trial_idx', 'condition',
        #      'event_type', 'event_idx'])
        result_df.drop(columns=['frame'], inplace=True)

        # Merge in behavior results
        # behav_df = pool.dataframes.behavior.behavior_df([run])
        # result_df = (pool.dataframes.smart_merge(result_df, behav_df,
        #                                          how='left')
        #              .set_index('error', append=True)
        #              .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
        #                               'condition', 'error', 'event_type',
        #                               'event_idx'])
        #              .sort_index()
        #              )

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

        # Merge in behavior results
        # behav_df = pool.dataframes.behavior.behavior_df([run])
        # result_df = (pool.dataframes.smart_merge(result_df, behav_df,
        #                                          how='left')
        #              .set_index(['condition', 'error'], append=True)
        #              .sort_index()
        #              )

        analysis = 'trialdf_frames_{}'.format(
            'inactmask' if inactivity_mask else 'noinactmask')

        out[analysis] = result_df

        return out

    def aligned_events(
            self, run, trigger, threshold, xmask, inactivity_mask):
        """
        Determine event times aligned to other (other than stimulus) events.

        Parameters
        ----------
        run : Run
        trigger : {'punishment', 'reward', 'lickbout'}
            Event to trigger PSTH on.
        threshold : float
            Classifier cutoff probability.
        xmask : bool
            If True, only allow one event (across types) per time bin.
        inactivity_mask : bool
            If True, enforce that all events are during times of inactivity.

        Note
        ----
        Individual events will appear in this DataFrame multiple times!
        Events may show up both as being after a triggering event and before
        the next one.

        """
        out = {}

        t2p = run.trace2p()

        if trigger == 'reward':
            onsets = t2p.reward()
            # There are 0s in place of un-rewarded trials.
            onsets = onsets[onsets > 0]
        elif trigger == 'punishment':
            onsets = t2p.punishment()
            # There are 0s in place of un-punished trials.
            onsets = onsets[onsets > 0]
        elif trigger == 'lickbout':
            onsets = t2p.lickbout()

        fr = t2p.framerate
        pre_fr = int(np.ceil(PRE_S * fr))
        post_fr = int(np.ceil(POST_S * fr))

        events = pool.dataframes.reactivation.events_df(
            [run], threshold, xmask=xmask,
            inactivity_mask=inactivity_mask)

        result = [pd.DataFrame({'trigger_idx': [], 'event_type': [],
                                'frame': [], 'time': []})]
        for trigger_idx, onset in enumerate(onsets):

            trigger_events = events.loc[
                (events.frame >= (onset - pre_fr)) &
                (events.frame < (onset + post_fr))].copy()
            trigger_events['frame'] -= onset
            trigger_events['time'] = trigger_events.frame / fr
            trigger_events['trigger_idx'] = trigger_idx

            result.append(trigger_events)

        result_df = (pd
                     .concat(result, axis=0)
                     .loc[:, ['trigger_idx', 'event_type', 'time']]
                     .sort_index()
                     )

        analysis = 'trialdf_{}_events_{}_{}_{}'.format(
            trigger,
            threshold,
            'xmask' if xmask else 'noxmask',
            'inactmask' if inactivity_mask else 'noinactmask')

        out[analysis] = result_df

        return out

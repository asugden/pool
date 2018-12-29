"""Trial dataframes."""
import numpy as np
import pandas as pd

import pool

from . import base


class TrialDf(base.AnalysisBase):
    """Trial-aligned reactivation dataframes."""

    requires = ['classifier']
    sets = ['trialdf_classifier'] + \
        ['trialdf_events_0.1_xmask_inactmask',
         'trialdf_events_0.1_xmask_noinactmask',
         'trialdf_events_0.1_noxmask_inactmask',
         'trialdf_events_0.1_noxmask_noinactmask'] + \
        ['trialdf_frames_inactmask',
         'trialdf_frames_noinactmask']

    across = 'run'
    updated = '1812273'

    def run(self, run):
        """Run everything."""
        out = {}

        # Classifier probability trace across trials.
        out.update(self.classifier(run))

        # Events relative to stimulus time
        for xmask in [True, False]:
            for inactivity_mask in [True, False]:
                for threshold in [0.1]:
                    out.update(self.events(
                        run, threshold=threshold, xmask=xmask,
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
        prev_onset_pad_s = 2.5
        next_onset_pad_s = 0.1
        c2p = run.classify2p()
        t2p = run.trace2p()

        classifier_results = c2p.results()
        all_onsets = t2p.csonsets()
        conditions = t2p.conditions()
        replay_types = pool.config.stimuli()

        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)
        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)

        fr = t2p.framerate

        next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
        prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

        result = [pd.DataFrame()]
        for trial_idx, (onset, next_onset, prev_onset, cond) in enumerate(
                zip(all_onsets, next_onsets, prev_onsets, conditions)):

            start_fr = prev_onset + prev_onset_pad_fr
            end_fr = next_onset - next_onset_pad_fr
            pre_fr = onset - start_fr

            trial_result = [pd.DataFrame()]
            for replay_type in replay_types:
                trial_replay_result = classifier_results[replay_type][
                    start_fr:end_fr - 1]
                time = (np.arange(len(trial_replay_result)) - pre_fr) / fr

                index = pd.MultiIndex.from_product(
                    [[run.mouse], [run.date], [run.run], [trial_idx], [cond],
                     time],
                    names=['mouse', 'date', 'run', 'trial_idx', 'condition',
                           'time'])
                trial_result.append(
                    pd.Series(trial_replay_result, index=index,
                              name=replay_type))
            result.append(pd.concat(trial_result, axis=1))
        final_result = pd.concat(result, axis=0)

        # Merge in behavior results
        behav_df = pool.dataframes.behavior.behavior_df([run])
        final_result = (pool.dataframes.smart_merge(final_result, behav_df,
                                                    how='left')
                        .set_index('error', append=True)
                        .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                         'condition', 'error', 'time'])
                        )

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
        Individual vents will appear in this dataframe multiple times!
        Events will show up both as being before after a stimulus and before
        the next one.

        """
        out = {}

        # These MUST exactly match the frames parameters below.
        prev_onset_pad_s = 2.5
        next_onset_pad_s = 0.1

        t2p = run.trace2p()

        all_onsets = t2p.csonsets()
        conditions = t2p.conditions()

        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

        fr = t2p.framerate
        next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
        prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

        events = pool.dataframes.reactivation.events_df(
            [run], threshold, xmask=xmask,
            inactivity_mask=inactivity_mask)

        result = [pd.DataFrame()]
        for trial_idx, (onset, next_onset, prev_onset, cond) in enumerate(zip(
                all_onsets, next_onsets, prev_onsets, conditions)):

            trial_events = events.loc[
                (events.frame >= (prev_onset + prev_onset_pad_fr)) &
                (events.frame < (next_onset - next_onset_pad_fr))].copy()
            trial_events -= onset
            trial_events['time'] = trial_events.frame / fr

            # add in trial_idx, condition
            trial_events = pd.concat(
                [trial_events], keys=[trial_idx], names=['trial_idx'])
            trial_events = pd.concat(
                [trial_events], keys=[cond], names=['condition'])

            result.append(trial_events)

        result_df = pd.concat(result, axis=0)
        result_df = result_df.reorder_levels(
            ['mouse', 'date', 'run', 'trial_idx', 'condition',
             'event_type', 'event_idx'])
        result_df.drop(columns=['frame'], inplace=True)

        # Merge in behavior results
        behav_df = pool.dataframes.behavior.behavior_df([run])
        result_df = (pool.dataframes.smart_merge(result_df, behav_df,
                                                 how='left')
                     .set_index('error', append=True)
                     .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                      'condition', 'error', 'event_type',
                                      'event_idx'])
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
        # These MUST exactly match the events parameters above.
        prev_onset_pad_s = 2.5
        next_onset_pad_s = 0.1

        t2p = run.trace2p()

        all_onsets = t2p.csonsets()

        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

        fr = t2p.framerate
        next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
        prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

        frames = pool.dataframes.imaging.frames_df([run], inactivity_mask)

        result = [pd.DataFrame()]
        for trial_idx, (onset, next_onset, prev_onset) in \
                enumerate(zip(all_onsets, next_onsets, prev_onsets)):

            trial_frames = frames.loc[
                (frames.frame >= (prev_onset + prev_onset_pad_fr)) &
                (frames.frame < (next_onset - next_onset_pad_fr))].copy()
            trial_frames.frame -= onset
            trial_frames['time'] = \
                trial_frames.frame * trial_frames.frame_period

            # add in trial_idx
            trial_frames = pd.concat(
                [trial_frames], keys=[trial_idx], names=['trial_idx'])

            result.append(trial_frames)

        result_df = pd.concat(result, axis=0)
        result_df = result_df.reorder_levels(
            ['mouse', 'date', 'run', 'trial_idx'])

        # Merge in behavior results
        behav_df = pool.dataframes.behavior.behavior_df([run])
        result_df = (pool.dataframes.smart_merge(result_df, behav_df,
                                                 how='left')
                     .set_index(['condition', 'error'], append=True)
                     )

        analysis = 'trialdf_frames_{}'.format(
            'inactmask' if inactivity_mask else 'noinactmask')

        out[analysis] = result_df

        return out

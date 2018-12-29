"""Helper functions for working with DataFrames."""
import pandas as pd

from .. import config


def smart_merge(df1, df2, how='left'):
    """
    Auto-merge on all shared indices/columns.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to merge.
    how : {'left', 'right', 'inner', 'outer'}
        Type of merge to perform. See pd.merge for details.

    Returns
    -------
    pd.DataFrame

    """
    first_idxs = df1.index.names
    first_columns = df1.columns
    first_all_keys = set(first_columns).union(first_idxs)

    second_idxs = df2.index.names
    second_columns = df2.columns
    second_all_keys = set(second_columns).union(second_idxs)

    shared_keys = list(first_all_keys.intersection(second_all_keys))

    df_merge = pd.merge(
        df1.reset_index(),
        df2.reset_index(),
        on=shared_keys, how=how).reset_index(drop=True)

    if how == 'left':
        df_merge = df_merge.set_index(first_idxs)
    if how == 'right':
        df_merge = df_merge.set_index(second_idxs)
    if how in ['inner', 'outer']:
        # Think this makes sense, change if needed.
        df_merge = df_merge.set_index(shared_keys)

    return df_merge.sort_index()


def bin_events(events, frames, edges, bin_labels):
    """
    Bin events and imaging frames, returning a merged DataFrame.

    Parameters
    ----------
    events : pd.DataFrame
    frames : pd.DataFrame
    edges : sequence of float
        Edges of bins, left-inclusive.
    bin_labels : sequence of str
        Names for bins.

    """
    events_gb = (events
                 .assign(time_cat=pd.cut(events.time, edges, labels=bin_labels,
                                         right=False))
                 .groupby(['mouse', 'date', 'run', 'condition', 'error',
                           'event_type', 'time_cat'])
                 .count()
                 .dropna()
                 .rename(columns={'time': 'events'})
                 )

    frames_gb = (frames
                 .drop(columns=['frame'])
                 .assign(time_cat=pd.cut(frames.time, edges, labels=bin_labels,
                                         right=False))
                 .groupby(['mouse', 'date', 'run', 'condition', 'error',
                           'time_cat', 'frame_period'])
                 .count()
                 .dropna()
                 .reset_index('frame_period')
                 .rename(columns={'time': 'frames'})
                 )

    # There has to be a better way to do this (merge?)
    #  There is: pivot on event_type, to add 3 columns with number of events
    #     per trial then merge with frame df and pivot back.
    # Expand across event types so that the merge will add in empty values
    all_times = []
    for event_type in config.stimuli():
        all_times.append(frames_gb
                         .assign(event_type=event_type)
                         .set_index('event_type', append=True))
    frames_gb = pd.concat(all_times, axis=0)

    result = (pd
              .merge(events_gb, frames_gb, how='right',
                     on=['mouse', 'date', 'run', 'time_cat', 'event_type',
                         'condition', 'error'])
              .reset_index(['time_cat', 'event_type', 'condition', 'error'])
              )

    # Add in 0's
    def fill_values(df):
        df['condition'] = \
            df['condition'].fillna(method='ffill').fillna(method='bfill')
        df['error'] = df['error'].fillna(method='ffill').fillna(method='bfill')
        df['events'] = df['events'].fillna(0)
        return df

    result = (result
              .groupby(['mouse', 'date', 'run'])
              .apply(fill_values))

    # Reset error to a boolean
    result['error'] = result['error'].astype('bool')

    result['event_rate'] = \
        result.events / (result.frame_period * result.frames)

    result = (result
              .reset_index()
              .set_index(['mouse', 'date', 'run', 'event_type', 'condition',
                          'error', 'time_cat'])
              .sort_index()
              )

    return result

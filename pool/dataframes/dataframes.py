"""Helper functions for working with DataFrames."""
import pandas as pd

from .. import config


def careful_first(series):
    """
    Use as a groupby aggregate function to ensure that all values are the same.

    Returns the first value after making sure that they are all the same.

    Parameters
    ----------
    series : pd.Series

    """
    assert series.unique() <= 1
    return series[0]


def smart_merge(df1, df2, how='left', sort=True):
    """
    Auto-merge on all shared indices/columns.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to merge.
    how : {'left', 'right', 'inner', 'outer'}
        Type of merge to perform. See pd.merge for details.
    sort : boolean
        If True, sort the output.

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

    if sort:
        df_merge = df_merge.sort_index()

    return df_merge


def bin_events(events, edges, labels=None, time_col='time'):
    """
    Bin event times into categories and converted to counts per bin.

    Parameters
    ----------
    events : pd.DataFrame
        Should have at least one column of event times. All other columns will
        be used for grouping.
    edges : sequence
        Edges of bins. Left-inclusive.
    labels : sequence of str, optional
        Names for the bins. len(labels) == len(edges) - 1
    time_col : str
        The name of the column containing the times to bin.

    Returns
    -------
    pd.DataFrame
        Binned and counted DataFrame. All previous indices and columns are now
        in the index, plus 'bin' which contains the bin label, and the previous
        'time_col' column is now named 'counts'.

    """
    if labels is None:
        labels = [str(x) for x in edges[:-1]]

    all_columns = events.index.names + list(events.columns)
    all_columns = [col for col in all_columns if col not in [None, time_col]]

    result = (events
              .assign(bin=pd.cut(events[time_col], edges, labels=labels,
                                 right=False))
              .groupby(all_columns + ['bin'])
              .count()
              .dropna()
              .rename(columns={time_col: 'count'})
              )

    return result


def event_rate(events, frames, event_label_col=None):
    """Calculate event rates from a DataFrame of events and frame times.

    All columns in events and frames should match and will be used to
    calculate the rates. DataFrame should already be binned, such that there
    is a column labeled 'count' that is the number of events/frames in each
    bin.

    Parameters
    ----------
    events : pd.DataFrame
    frames :pd.DataFrame
    event_label_col : str, optional
        Optionally, calculate event rate for each "flavor" or event
        independently. If not None, pivot events on this column to calculate
        a rate for each type.

    Returns
    -------
    pd.DataFrame

    """
    if event_label_col is not None:
        # Convert to wide form if there multiple 'flavors' of events
        # Allows us to can add in 0's where necessary
        events = events.unstack(event_label_col)
        events.columns = events.columns.droplevel()

    # Convert frame counts to time
    tmp_df = frames.reset_index('frame_period')
    frames_series = (tmp_df['count'] * tmp_df['frame_period'])

    # Expand back out to correctly add 0's
    events = events.reindex(frames_series.index).fillna(0)

    # Calc rate
    rate_df = events.div(frames_series, axis=0)

    if event_label_col is not None:
        # Convert back to long form
        rate_df = rate_df.stack().to_frame('event_rate')

    return rate_df

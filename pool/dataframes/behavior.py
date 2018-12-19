import numpy as np
import pandas as pd


def behavior_df(runs):
    """Build a dataframe of all behavior results.

    Parameters
    ----------
    runs : RunSorter or list of Runs

    Returns
    -------
    pd.DataFrame

    """
    result_list = []
    for run in runs:
        t2p = run.trace2p()

        conditions = t2p.conditions()
        # Generalize outcomes to it works for all stims?
        # outcomes = t2p.outcomes()
        errors = t2p.errors()

        index = pd.MultiIndex.from_product([
            [run.mouse], [run.date], [run.run], np.arange(len(conditions))],
            names=['mouse', 'date', 'run', 'trial_idx'])

        result_list.append(pd.DataFrame(
            {'conditions': conditions, 'errors': errors}, index=index))

    result = pd.concat(result_list, axis=0)

    return result

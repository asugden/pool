import numpy as np
import flow.metadata.sorters
import pool.config

def trials(runs, cs, start_s=0, end_s=None, trace_type='dff', cutoff_before_lick_ms=-1,
           error_trials=-1, baseline=None, pavlovian=True, end_relative=0):
    """
    Get all training trials.

    Parameters
    ----------
    runs : RunSorter or Date from which the non-sated training runs will be extracted
    cs : string or list of strings
            CS-type to return traces of, Should be one of values returned by
            t2p.cses().
    start_s : float
        Time before stim to include, in seconds.
        For backward compatibility, can also be arg dict.
    end_s : float
        Time after stim to include, in seconds. If None, will be set to the
        stimulus length + end_relative.
    trace_type : {'deconvolved', 'raw', 'dff'}
        Type of trace to return.
    cutoff_before_lick_ms : int
        Exclude all time around licks by adding NaN's this many ms before
        the first lick after the stim.
    error_trials : {-1, 0, 1}
        -1 is all trials, 0 is only correct trials, 1 is error trials
    baseline : tuple of 2 ints, optional
        Use this interval (in seconds) as a baseline to subtract off from
        all traces each trial.
    pavlovian : bool
        If true, include pavlovian trials with cs plus trials.
    end_relative : float
        The time relative to the stimulus length in seconds

    Returns
    -------
    ndarray
        ncells x frames x nstimuli/onsets

    """

    # Convert date to runsorter of training sessions
    if isinstance(runs, flow.metadata.sorters.Run):
        runs = [runs]
    elif isinstance(runs, flow.metadata.sorters.Date):
        runs = runs.runs(run_types='training', tags='hungry')

    # Convert a string cs into an iterable list
    if len(cs) == 0:
        cses = pool.config.stimuli()
    elif not isinstance(cs, list):
        cses = [cs]
    else:
        cses = cs

    alltrs = []
    for run in runs:
        t2p = run.trace2p()

        if end_s is None:
            end_s = t2p.stimulus_length + end_relative

        for cs in cses:
            # ncells, frames, nstimuli/onsets
            trs = t2p.cstraces(cs, start_s=start_s, end_s=end_s, trace_type=trace_type,
                               cutoff_before_lick_ms=cutoff_before_lick_ms, errortrials=error_trials,
                               baseline=baseline)
            if cs == 'plus' and pavlovian:
                pavs = t2p.cstraces('pavlovian', start_s=start_s, end_s=end_s, trace_type=trace_type,
                                    cutoff_before_lick_ms=cutoff_before_lick_ms, errortrials=error_trials,
                                    baseline=baseline)
                trs = np.concatenate([trs, pavs], axis=2)

            if len(alltrs) == 0:
                alltrs = trs
            else:
                alltrs = np.concatenate([alltrs, trs], axis=2)

    return alltrs
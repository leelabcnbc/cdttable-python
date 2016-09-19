import numpy as np
from joblib import Parallel, delayed
from .schema import validate, EventSplittingParamsSchema


def _check_input(codes: np.ndarray, times: np.ndarray) -> tuple:
    assert isinstance(codes, np.ndarray) and isinstance(times, np.ndarray)
    assert codes.shape == times.shape
    assert codes.ndim == 1

    times = times.astype(np.float64, copy=False, casting='safe')
    codes = codes.astype(np.int16, copy=False, casting='safe')
    return codes, times


def _get_times_one_sub_trial(codes: np.ndarray, times: np.ndarray, trial_info: dict) -> tuple:
    start_time = times[np.flatnonzero(codes == trial_info['start_code'])[0]]
    if 'end_time' in trial_info:
        end_time = start_time + trial_info['end_time']
    elif 'end_code' in trial_info:
        end_time = times[np.flatnonzero(codes == trial_info['end_code'])[0]]
    else:
        raise RuntimeError('unsupported mode for finding end of subtrial!')
    return start_time, end_time


def _get_times_subtrials(codes: np.ndarray, times: np.ndarray, params: dict) -> tuple:
    # find (abs) subtrial start/stop times.
    start_times = []
    end_times = []
    for trial_info in params:
        start_time, end_time = _get_times_one_sub_trial(codes, times, trial_info)
        start_times.append(start_time)
        end_times.append(end_times)
    start_times = np.asarray(start_times)
    end_times = np.asarray(end_times)
    assert np.all(start_times <= end_times)
    return start_times, end_times


def _get_times_trial(codes: np.ndarray, times: np.ndarray,
                     start_times: np.ndarray, end_times: np.ndarray, params: dict) -> tuple:
    # find trial start/end times
    if 'trial_start_code' in params:
        trial_start_time = times[np.flatnonzero(codes == params['trial_start_code'])[0]]
        if 'trial_end_code' in params:
            trial_end_time = times[np.flatnonzero(codes == params['trial_end_code'])[0]]
        elif 'trial_end_time' in params:
            trial_end_time = trial_start_time + params['trial_end_time']
        else:
            raise RuntimeError('unsupported mode for finding end of trial!')
    else:
        trial_start_time = start_times[0]
        trial_end_time = end_times[-1]

    assert trial_start_time <= trial_end_time
    trial_start_time -= params['margin_before']
    trial_end_time += params['margin_after']
    return trial_start_time, trial_end_time


def _split_events_per_trial(codes: np.ndarray, times: np.ndarray,
                            trial_to_condition_func, params: dict) -> dict:
    """roughly a rewrite of import_one_trial_startstoptime

    Parameters
    ----------
    codes
    times
    trial_to_condition_func
    params

    Returns
    -------

    """
    codes, times = _check_input(codes, times)
    cnd_number = trial_to_condition_func(codes).astype(np.int16, casting='safe', copy=False)
    assert np.isscalar(cnd_number)

    start_times, end_times = _get_times_subtrials(codes, times, params['subtrials'])
    trial_start_time, trial_end_time = _get_times_trial(codes, times, start_times, end_times, params)

    assert np.all(trial_start_time <= start_times)
    assert np.all(trial_end_time >= end_times)

    event_code_idx = np.logical_and(times >= trial_start_time, times <= trial_end_time)

    return {
        'start_time': start_times,
        'end_time': end_times,
        'trial_start_time': trial_start_time,
        'trial_end_time': trial_end_time,
        'event_times': times[event_code_idx],
        'event_codes': codes[event_code_idx],
        'condition_number': cnd_number
    }


def split_events(event_data: dict, event_splitting_params: dict, debug=False) -> dict:
    """extract event codes according to event splitting params.
    basically, this code replicates half the of `+cdttable/import_one_trial.m` in the original matlab package.
    the other half should go to the splitter.

    Parameters
    ----------
    event_data
    event_splitting_params
    debug: boolean
        only used in unit testing.

    Returns
    -------

    """
    if debug:
        assert validate(EventSplittingParamsSchema.get_schema(), event_splitting_params)

    # check that event_data is of good form.
    assert {'event_codes', 'event_times'} == set(event_data.keys())
    assert len(event_data['event_codes']) == len(event_data['event_times'])

    trial_to_condition_func = eval(event_splitting_params['trial_to_condition_func'], {}, {})

    # no memmaping, since trials are usually short.
    pool = Parallel(n_jobs=-1, max_nbytes=None)
    pool(delayed(_split_events_per_trial)(codes, times,
                                          trial_to_condition_func,
                                          event_splitting_params) for codes, times in zip(event_data['event_codes'],
                                                                                          event_data['event_times']))

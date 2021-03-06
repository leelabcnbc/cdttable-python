import numpy as np
from joblib import Parallel, delayed


def _check_input(codes, times) -> tuple:
    # I don't want to work with subclass of ndarray, as subclasses actually have some undesired effects.
    codes = np.asarray(codes)
    times = np.asarray(times)
    assert codes.shape == times.shape
    assert codes.ndim == 1

    times = times.astype(np.float64, copy=False, casting='safe')
    codes = codes.astype(np.int16, copy=False, casting='safe')
    return codes, times


def _check_output(result: dict) -> None:
    """

    Parameters
    ----------
    result

    Returns
    -------

    """
    float64_set = {'start_times', 'end_times',
                   'trial_start_time', 'trial_end_time', 'trial_length'}
    for x in float64_set:
        assert result[x].dtype == np.float64
    assert result['condition_number'].dtype == np.int16
    for x in result['event_times']:
        assert x.dtype == np.float64
    for x in result['event_codes']:
        assert x.dtype == np.int16


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
        end_times.append(end_time)
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


def _split_events_per_trial(t_idx, codes: np.ndarray, times: np.ndarray, params: dict) -> dict:
    """roughly a rewrite of import_one_trial_startstoptime

    Parameters
    ----------
    codes
    times
    params

    Returns
    -------

    """
    codes, times = _check_input(codes, times)
    trial_to_condition_func = eval(params['trial_to_condition_func'], {}, {})
    cnd_number = np.int16(trial_to_condition_func(codes, t_idx))
    assert np.isscalar(cnd_number)

    start_times, end_times = _get_times_subtrials(codes, times, params['subtrials'])
    trial_start_time, trial_end_time = _get_times_trial(codes, times, start_times, end_times, params)

    # this is due to ill design of trial window and subtrial window due to human error.
    assert np.all(trial_start_time <= start_times)
    assert np.all(trial_end_time >= end_times)

    event_code_idx = np.logical_and(times >= trial_start_time, times <= trial_end_time)

    return {
        'start_times': start_times,  # absolute
        'end_times': end_times,  # absolute
        'trial_start_time': trial_start_time,
        'trial_end_time': trial_end_time,
        'event_times': times[event_code_idx],
        'event_codes': codes[event_code_idx],
        'condition_number': cnd_number
    }


def _assemble_result(split_result, n_trial):
    fields_to_be_ndarray = {
        'start_times', 'end_times', 'trial_start_time', 'trial_end_time', 'condition_number'
    }

    result = {}
    for field in fields_to_be_ndarray:
        result[field] = np.asarray([x[field] for x in split_result])

    result['trial_length'] = result['trial_end_time'] - result['trial_start_time']

    # normalize time w.r.t trial start.
    result['start_times'] -= result['trial_start_time'][:, np.newaxis]
    result['end_times'] -= result['trial_start_time'][:, np.newaxis]

    # create object arrays.
    event_times = np.empty(n_trial, dtype=np.object_)
    event_codes = np.empty(n_trial, dtype=np.object_)

    for idx in range(n_trial):
        event_times[idx] = split_result[idx]['event_times'] - result['trial_start_time'][idx]
        event_codes[idx] = split_result[idx]['event_codes']

    result['event_times'] = event_times
    result['event_codes'] = event_codes

    return result


def split_events(event_data: dict, event_splitting_params: dict, n_jobs=-1) -> dict:
    """extract event codes according to event splitting params.
    basically, this code replicates half the of `+cdttable/import_one_trial.m` in the original matlab package.
    the other half should go to the splitter.

    Parameters
    ----------
    event_data
    event_splitting_params
    n_jobs

    Returns
    -------

    """
    # check that event_data is of good form.
    assert {'event_codes', 'event_times'} == set(event_data.keys())
    n_trial = len(event_data['event_codes'])
    assert n_trial == len(event_data['event_times'])
    assert n_trial >= 1

    # no memmaping, since trials are usually short.
    pool = Parallel(n_jobs=n_jobs, max_nbytes=None)
    split_result = pool(
        delayed(_split_events_per_trial)(t_idx, codes, times, event_splitting_params) for t_idx, (codes, times) in
        enumerate(zip(event_data['event_codes'],
                      event_data['event_times'])))

    result = _assemble_result(split_result, n_trial)
    _check_output(result)

    return result

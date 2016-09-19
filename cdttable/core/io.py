from collections import OrderedDict
from .schema import DataMetaJSL, validate
from joblib import Parallel, delayed
import numpy as np


def unique_rows(a):
    assert a.ndim == 2
    # from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    if a.size > 0:
        result = np.vstack({tuple(row) for row in a})
    else:
        result = a
    return result


def import_sessions(session_gen, import_params):
    """import multiple sesssions

    Parameters
    ----------
    session_gen
    import_params

    Returns
    -------
    a generator object for getting processed sessions. Thus it can be used in a very
    flexible way. No unnecessary memory consumption.
    """
    return (import_one_session(session, import_params) for session in session_gen)


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
    assert isinstance(codes, np.ndarray) and isinstance(times, np.ndarray)
    assert codes.shape == times.shape
    assert codes.ndim == 1

    times = times.astype(np.float64, copy=False, casting='safe')
    codes = codes.astype(np.int16, copy=False, casting='safe')
    cnd_number = trial_to_condition_func(codes).astype(np.int16, casting='safe', copy=False)

    # find (abs) subtrial start/stop times.
    start_times = []
    end_times = []
    for trial_info in params['subtrials']:
        start_time = times[np.flatnonzero(codes == trial_info['start_code'])[0]]
        if 'end_time' in trial_info:
            end_time = start_time + trial_info['end_time']
        elif 'end_code' in trial_info:
            end_time = times[np.flatnonzero(codes == trial_info['end_code'])[0]]
        else:
            raise RuntimeError('unsupported mode for finding end of subtrial!')
        start_times.append(start_time)
        end_times.append(end_time)
    start_times = np.asarray(start_times)
    end_times = np.asarray(end_times)

    assert np.all(start_times <= end_times)

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
    assert np.all(trial_start_time <= start_times)
    assert np.all(trial_end_time >= end_times)

    event_code_idx = np.logical_and(times >= trial_start_time, times <= trial_end_time)

    return {
        'start_time': start_times,
        'stop_time': end_times,
        'trial_start_time': trial_start_time,
        'trial_stop_time': trial_end_time,
        'event_times': times[event_code_idx],
        'event_codes': codes[event_code_idx],
    }


def split_events(event_data: dict, event_splitting_params: dict) -> dict:
    """extract event codes according to event splitting params.
    basically, this code replicates half the of `+cdttable/import_one_trial.m` in the original matlab package.
    the other half should go to the splitter.

    Parameters
    ----------
    event_data
    event_splitting_params

    Returns
    -------

    """
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


def import_one_session(session, import_params):
    """

    Parameters
    ----------
    session
    import_params

    Returns
    -------
    some (ordered?) dict object having all columns ready to be saved to hdf5 or mat
    """
    # TODO: check import_params is good.

    event_splitting_result = split_events(session['event_data'], import_params['event_splitting_params'])
    # first let's work on the time markers events.

    # work out the start and stop times of each trial.
    start_times = None
    stop_times = None

    processed_session_dict = OrderedDict()

    for table_name, table_data_raw in session['data'].items():
        # extract the json describing how this piece of data looks like.
        data_meta_this = import_params['data_meta'][table_name]
        processed_session_dict[table_name] = import_one_data_table(table_data_raw, data_meta_this,
                                                                   event_splitting_result,
                                                                   import_params)


def import_one_data_table(data_raw, data_meta, event_info, import_params):
    assert validate(DataMetaJSL.get_schema(), data_meta)
    data_type = data_meta['type']
    # do preparation for all data types.
    if data_type == 'point_process_1d':
        splitter = splitter_point_process
    else:
        raise ValueError('not supported data type: {}'.format(data_type))

    if not data_meta['split']:
        # ok, split the data myself.
        splitted_data = splitter(data_raw, data_meta, event_info, import_params)
    else:
        splitted_data = data_raw

    result = OrderedDict()
    # then combine them.
    if not data_meta['location_columns']:
        # just a np array always of shape (n_trial, ... )
        values = splitted_data['values']
    else:
        # the results' shape[0] is number of all occurring trials and locations.
        locations = np.concatenate([x['locations'] for x in splitted_data], axis=0)
        values = _concatenate_trials_with_locations(splitted_data, data_type)
        result[data_meta['values_name']] = values

    return splitted_data


def _concatenate_trials_with_locations(splitted_data, data_type):
    table_length = sum([len(x) for x in splitted_data])

    if data_type == 'fixed':
        # easy, simply combine them.
        values = np.concatenate([x for x in splitted_data], axis=0)
    else:
        # use a object type.
        dtype_to_use = np.object_
        values = np.empty(shape=(table_length,), dtype=np.object_)
        # fill in.


def splitter_point_process(data_raw, data_meta, event_info, import_params):
    timestamps = data_raw['values']
    assert timestamps.ndim == 1

    # no matter whether timestamps is empty or not, there's always possibility to get empty stuff.
    num_trial = event_info['num_trial']
    if data_meta['location_columns']:
        locations = np.concatenate([data_raw[x] for x in data_meta['location_columns']], axis=1)
    else:
        locations = None

    # do a par for loop, over each trial
    timestamps_per_trial = Parallel(n_jobs=import_params['n_jobs'],
                                    verbose=5)(delayed(_splitter_pp_inner)(timestamps,
                                                                           locations,
                                                                           event_info['start_times'],
                                                                           event_info['stop_times'],
                                                                           idx) for idx in range(num_trial))

    return timestamps_per_trial


def _splitter_pp_inner(timestamps, locations, start_times, stop_times, idx):
    start_time = start_times[idx]
    stop_time = stop_times[idx]
    valid_idx = np.logical_and(timestamps > start_time, timestamps < stop_time)
    spike_times = timestamps[valid_idx]
    if locations is not None:
        # we need to sort all valid indices according to their locations.
        locations_this_trial = locations[valid_idx]
        locations_this_trial_unique = unique_rows(locations_this_trial)
        spike_times_list = []
        for location_this_trial_unique_this in locations_this_trial_unique:
            spikes_this_location = spike_times[np.all(location_this_trial_unique_this == locations_this_trial, axis=1)]
            spike_times_list.append(spikes_this_location)
        return {'locations': locations_this_trial,
                'values': spike_times_list}
    else:
        return {'values': spike_times}


def export_sessions(processed_session_gen, export_format, convert_params):
    # let's do this sequentially.
    if export_format == 'mat':
        processor = _export_sessions_mat
    elif export_format == 'hdf5':
        processor = _export_sessions_hdf5
    else:
        raise ValueError('unsupported export format! must be mat or hdf5!')
    processor(processed_session_gen, convert_params)


def _export_sessions_mat():
    pass


def _export_sessions_hdf5():
    pass

from collections import OrderedDict
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


def import_one_data_table(data_raw, data_meta, event_info, import_params):
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

import numpy as np
from . import root_package_spec


def _import_one_data_table_split(data_raw, data_meta, event_info):
    # optionally, we should split the data.
    if 'splitter' in data_meta:
        code_to_run = 'from {pkg}.splitter.{splitter_name} import run'.format(pkg=root_package_spec,
                                                                              splitter_name=data_meta['splitter'])
        locals_query = {}
        exec(code_to_run, {}, locals_query)
        spltter_instance = locals_query['run']
        data_ready_to_do = spltter_instance(data_raw, event_info, data_meta['splitter_params'])
    else:
        data_ready_to_do = data_raw

    return data_ready_to_do


def _import_one_data_table_trial_level(data_ready_to_do, data_meta, event_info):
    if data_meta['with_timestamp']:
        timestamp_type = data_meta['timestamp_type']
        timestamp_column = data_meta['timestamp_column']
        trial_start_time = event_info['trial_start_time']
        if timestamp_type == 'absolute':
            start_time_absolute = data_ready_to_do[timestamp_column]
            start_time_relative = start_time_absolute - trial_start_time
        elif timestamp_type == 'relative':
            start_time_relative = data_ready_to_do[timestamp_column]
            start_time_absolute = start_time_relative + trial_start_time
        else:
            raise ValueError('no such timestamp type supported!')
        assert start_time_absolute.shape == start_time_relative.shape == trial_start_time.shape
        result = {
            'start_time_absolute': start_time_absolute,
            'start_time_relative': start_time_relative,
        }
    else:
        result = {}
    return result


def _import_one_data_table_location_level_check_input(data_ready_to_do, data_meta, n_trial):
    data_type = data_meta['type']

    # make sure same number as n_trial
    value_column = data_ready_to_do[data_meta['value_column']]
    assert len(value_column) == n_trial
    trial_iter = [value_column, ]
    for loc_column_name in data_meta['location_columns']:
        loc_column_this = data_ready_to_do[loc_column_name]
        assert len(loc_column_this) == n_trial
        trial_iter.append(loc_column_this)

    if data_type == 'fixed':
        if data_meta['location_columns']:
            data_shape = data_ready_to_do[data_meta['value_column']][0].shape[1:]  # this also works for scalar.
        else:
            data_shape = data_ready_to_do[data_meta['value_column']][0].shape
    else:
        # others are variable, and don't check shape at all.
        data_shape = None

    num_location_list = []
    for value_column_this_trial, *location_column_this_trial_list in zip(*trial_iter):
        if location_column_this_trial_list:
            # TODO: think about what happens when this is empty. (n_locations_this_trial=0)
            n_locations_this_trial = len(value_column_this_trial)
            for x in location_column_this_trial_list:
                assert x.shape == (n_locations_this_trial,)
            shape_prefix = (n_locations_this_trial,)
            num_location_list.append(n_locations_this_trial)
        else:
            shape_prefix = ()

        if data_type == 'fixed':
            assert value_column_this_trial.shape == shape_prefix + data_shape

    # return number of locations in each trial.
    return num_location_list


def _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info):
    """combine split data from different trials into several big columns.

    Parameters
    ----------
    data_ready_to_do
    data_meta
    event_info

    Returns
    -------

    """
    n_trial = len(event_info['event_codes'])
    value_column_name = data_meta['value_column']
    value_column_save_name = 'value_{}'.format(value_column_name)
    value_column = data_ready_to_do[value_column_name]
    num_location_list = _import_one_data_table_location_level_check_input(data_ready_to_do, data_meta, n_trial)
    # ok. now time to combine them.
    result = {}
    if data_meta['location_columns']:
        # this one is trivial, simply concatenate everything along first dim
        # let's first work on locations.
        trial_idx = np.concatenate([np.full(x, idx, dtype=np.int64) for idx, x in enumerate(num_location_list)])
        for idx, x in enumerate(data_meta['location_columns']):
            result['location_{}_{}'.format(idx, x)] = np.concatenate(data_ready_to_do[x], axis=0)
        result[value_column_save_name] = np.concatenate(value_column, axis=0)
    else:
        trial_idx = np.arange(n_trial, dtype=np.int64)
        # only work on values
        if data_meta['type'] == 'fixed':
            # so we can simply use asarray
            values_to_save = np.asarray(value_column)
        else:
            # explicitly create array first so it won't get converted into something else.
            values_to_save = np.empty(n_trial, dtype=np.object_)
            for t_idx, value_this in enumerate(value_column):
                values_to_save[t_idx] = value_this
        result[value_column_save_name] = values_to_save
        # ok time to deal with values.
    result['trial_idx'] = trial_idx

    return result


def import_one_data_table(data_raw, data_meta, event_info):
    data_ready_to_do = _import_one_data_table_split(data_raw, data_meta, event_info)
    # work out the trial_level stuff
    trial_level_data = _import_one_data_table_trial_level(data_ready_to_do, data_meta, event_info)
    # work out the things at location level.
    location_level_data = _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info)
    # no overlap.
    assert set(trial_level_data.keys()) & set(location_level_data.keys()) == set()

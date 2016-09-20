from joblib import Parallel, delayed
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
        if timestamp_type == 'absolute':
            start_time_absolute = data_ready_to_do[timestamp_column]
            start_time_relative = start_time_absolute - event_info['trial_start_time']
        else:
            start_time_relative = data_ready_to_do[timestamp_column]
            start_time_absolute = start_time_relative + event_info['trial_start_time']
        result = {
            'start_time_absolute': start_time_absolute,
            'start_time_relative': start_time_relative,
        }
    else:
        result = {}
    return result


def _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info):
    return {}


def import_one_data_table(data_raw, data_meta, event_info):
    data_ready_to_do = _import_one_data_table_split(data_raw, data_meta, event_info)
    # work out the trial_level stuff
    trial_level_data = _import_one_data_table_trial_level(data_ready_to_do, data_meta, event_info)
    # work out the things at location level.
    location_level_data = _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info)
    # no overlap.
    assert set(trial_level_data.keys()) & set(location_level_data.keys()) == set()

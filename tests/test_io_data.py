import unittest
from itertools import product
import numpy as np
from cdttable.core.io_data import (_import_one_data_table_trial_level,
                                   _import_one_data_table_location_level,
                                   import_one_data_table)
from cdttable import test_util


class MyTestCase(unittest.TestCase):
    def setUp(self):
        test_util.reseed(42)

    def test_import_one_data_table_trial_level(self):
        timestamp_type_list = ['absolute', 'relative', 'blablabla']
        for timestamp_type in timestamp_type_list:
            for n_trial in [1, 100, None]:
                if n_trial is None:
                    n_trial_real = test_util.rng_state.randint(1, 500 + 1)
                else:
                    n_trial_real = n_trial
                trial_start_time = test_util.generate_random_times(n_trial=n_trial_real)
                timestamp_column = test_util.generate_random_times(n_trial=n_trial_real)

                if timestamp_type == 'absolute':
                    timestamp_abs = timestamp_column.copy()
                    timestamp_rel = timestamp_column - trial_start_time
                else:
                    timestamp_rel = timestamp_column.copy()
                    timestamp_abs = trial_start_time + timestamp_rel
                colname = test_util.generate_random_column_name()
                data_meta = {'with_timestamp': True,
                             'timestamp_type': timestamp_type,
                             'timestamp_column': colname}
                data_ready_to_do = {colname: timestamp_column}
                event_info = {'trial_start_time': trial_start_time}

                with self.subTest(n_trial=n_trial_real, timestamp_type=timestamp_type, colname=colname):
                    if timestamp_type == 'blablabla':
                        with self.assertRaises(ValueError):
                            _import_one_data_table_trial_level(data_ready_to_do, data_meta, event_info)
                    else:
                        result = _import_one_data_table_trial_level(data_ready_to_do, data_meta, event_info)
                        self.assertEqual(result['start_time_absolute'].shape, (n_trial_real,))
                        self.assertEqual(result['start_time_relative'].shape, (n_trial_real,))
                        self.assertTrue(np.allclose(result['start_time_absolute'], timestamp_abs))
                        self.assertTrue(np.allclose(result['start_time_relative'], timestamp_rel))

    def test_import_one_data_table_trial_level_trivial(self):
        # first, test no time stamp
        result = _import_one_data_table_trial_level(None, {'with_timestamp': False}, None)
        self.assertEqual(result, {})

    def test_import_one_data_table_location_level_without_locations_fixed(self):
        ndim_list = [0, 1, 2, 3, 4, 5, None]

        n_trial_list = [1, 10, 100]

        for ndim, n_trial, dtype in product(ndim_list, n_trial_list, [np.uint8,
                                                                      np.uint16,
                                                                      np.uint32,
                                                                      np.uint64,
                                                                      np.int8,
                                                                      np.int16,
                                                                      np.int32,
                                                                      np.int64, np.float32, np.float64
                                                                      ]):
            shape_this = test_util.generate_random_shape(ndim)
            full_data = (test_util.rng_state.randn(n_trial, *shape_this) * 10000).astype(dtype)

            event_info = {'event_codes': np.empty(n_trial)}
            colname = test_util.generate_random_column_name()
            data_ready_to_do = {colname: [x.copy() for x in full_data]}
            data_meta = {'value_column': colname, 'location_columns': [], 'type': 'fixed'}

            ndim_real = len(shape_this)
            if ndim is not None:
                assert ndim == ndim_real
            with self.subTest(ndim=ndim_real, n_trial=n_trial, dtype=dtype):
                result = _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info)
                value_column_save_name = 'value_{}'.format(colname)
                self.assertEqual({value_column_save_name, 'trial_idx'}, set(result.keys()))
                self.assertTrue(np.array_equal(result[value_column_save_name], full_data))
                self.assertTrue(np.array_equal(result['trial_idx'], np.arange(n_trial)))

    def test_import_one_data_table_location_level_without_locations_variable_1d(self):
        n_trial_list = [1, 10, 100]

        for n_trial, dtype in product(n_trial_list, [np.uint8,
                                                     np.uint16,
                                                     np.uint32,
                                                     np.uint64,
                                                     np.int8,
                                                     np.int16,
                                                     np.int32,
                                                     np.int64, np.float32, np.float64
                                                     ]):
            # generate full data
            numbers_per_trial = test_util.rng_state.randint(1, 20, n_trial)
            num_to_make_empty = max(1, n_trial // 2)
            numbers_per_trial[test_util.rng_state.choice(n_trial, num_to_make_empty)] = 0  # make half of them empty.

            full_data = []
            for idx in range(n_trial):
                full_data_this = (test_util.rng_state.randn(numbers_per_trial[idx]) * 10000).astype(dtype)
                full_data.append(full_data_this)

            event_info = {'event_codes': np.empty(n_trial)}
            colname = test_util.generate_random_column_name()
            data_ready_to_do = {colname: [x.copy() for x in full_data]}
            data_meta = {'value_column': colname, 'location_columns': [], 'type': 'variable_1d'}
            with self.subTest(n_trial=n_trial, dtype=dtype):
                result = _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info)
                value_column_save_name = 'value_{}'.format(colname)
                self.assertEqual({value_column_save_name, 'trial_idx'}, set(result.keys()))
                self.assertEqual(result[value_column_save_name].shape, (n_trial,))
                self.assertTrue(np.array_equal(result['trial_idx'], np.arange(n_trial)))
                for a1, a2 in zip(result[value_column_save_name], full_data):
                    self.assertTrue(np.array_equal(a1, a2))

    def test_import_one_data_table_location_level_with_locations_fixed(self):
        ndim_list = [0, 1, 2, 3, 4, 5, None]

        n_trial_list = [1, 10, 100]

        for ndim, n_trial, dtype in product(ndim_list, n_trial_list, [np.uint8,
                                                                      np.uint16,
                                                                      np.uint32,
                                                                      np.uint64,
                                                                      np.int8,
                                                                      np.int16,
                                                                      np.int32,
                                                                      np.int64, np.float32, np.float64
                                                                      ]):
            shape_this = test_util.generate_random_shape(ndim)
            locations_per_trial = test_util.rng_state.randint(1, 20, n_trial)
            num_to_make_empty = max(1, n_trial // 2)
            locations_per_trial[test_util.rng_state.choice(n_trial, num_to_make_empty)] = 0  # make half of them empty.
            num_location_cols = test_util.rng_state.randint(1, 5)
            location_col_names = [test_util.generate_random_column_name() for _ in range(num_location_cols)]
            location_col_names_saved = ['location_{}_{}'.format(idx, nm) for idx, nm in enumerate(location_col_names)]

            full_data = (test_util.rng_state.randn(locations_per_trial.sum(), *shape_this) * 10000).astype(dtype)
            assert np.all(np.isfinite(full_data))
            # now time to get split data.
            split_data = []

            full_loc_data = []
            for loc_col_idx in range(num_location_cols):
                full_loc_data.append(test_util.rng_state.randint(-1000, 1000, locations_per_trial.sum()))

            split_loc_data = [[] for _ in range(num_location_cols)]
            finished = 0

            trial_index_true = []

            for idx, num_loc in enumerate(locations_per_trial):
                slice_this = slice(finished, finished + num_loc)
                split_data.append(full_data[slice_this].copy())
                # for each of location_col_data, work on it.
                for loc_col_idx in range(num_location_cols):
                    split_loc_data[loc_col_idx].append(full_loc_data[loc_col_idx][slice_this].copy())
                if num_loc != 0:
                    trial_index_true.append(np.full(num_loc, idx, dtype=np.int64))
                finished += num_loc
            if trial_index_true:
                trial_index_true = np.concatenate(trial_index_true, axis=0)
            else:
                trial_index_true = np.array([], dtype=np.int64)

            event_info = {'event_codes': np.empty(n_trial)}
            colname = test_util.generate_random_column_name()
            data_ready_to_do = {colname: split_data}
            data_meta = {'value_column': colname, 'location_columns': location_col_names, 'type': 'fixed'}

            for loc_col_idx in range(num_location_cols):
                data_ready_to_do[location_col_names[loc_col_idx]] = split_loc_data[loc_col_idx]

            ndim_real = len(shape_this)
            if ndim is not None:
                assert ndim == ndim_real
            with self.subTest(ndim=ndim_real, n_trial=n_trial, dtype=dtype, num_location_cols=num_location_cols,
                              num_to_make_empty=num_to_make_empty):
                result = _import_one_data_table_location_level(data_ready_to_do, data_meta, event_info)
                value_column_save_name = 'value_{}'.format(colname)
                self.assertEqual({value_column_save_name, 'trial_idx'} | set(location_col_names_saved),
                                 set(result.keys()))
                self.assertTrue(np.array_equal(result[value_column_save_name], full_data))
                self.assertTrue(np.array_equal(result['trial_idx'], trial_index_true))
                for idx, loc_col_name in enumerate(location_col_names_saved):
                    self.assertTrue(np.array_equal(result[loc_col_name], full_loc_data[idx]))


if __name__ == '__main__':
    unittest.main()

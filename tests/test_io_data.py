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


if __name__ == '__main__':
    unittest.main()

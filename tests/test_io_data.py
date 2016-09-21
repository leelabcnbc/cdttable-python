import unittest
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


if __name__ == '__main__':
    unittest.main()

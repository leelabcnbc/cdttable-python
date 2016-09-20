import unittest
from cdttable.core.io_event import (split_events)
from cdttable import test_util
import numpy as np
from itertools import product
from cdttable.core.schema import validate, EventSplittingParamsSchema


class MyTestCase(unittest.TestCase):
    def setUp(self):
        test_util.reseed(42)

    def test_split_event_all_codes(self):
        margin_before, margin_after = test_util.generate_margins()
        num_trial_list = [1, 50, 100, 150, None]
        num_pair_list = [1, 2, 3, 4, 5, None]
        using_all_codes_list = [True, False]
        for num_trial_, num_pair_, all_codes in product(num_trial_list, num_pair_list, using_all_codes_list):
            test_data = test_util.generate_event_data(margin_before, margin_after,
                                                      num_trial=num_trial_, all_codes=all_codes)
            if num_trial_ is not None:
                num_trial = num_trial_
            else:
                num_trial = len(test_data['event_codes'])

            if num_pair_ is not None:
                num_pair = num_pair_
            else:
                num_pair = len(test_data['start_codes'])

            with self.subTest(num_trial=num_trial, num_pair=num_pair, using_all_codes=all_codes):
                event_data = {
                    'event_codes': test_data['event_codes'],
                    'event_times': test_data['event_times']
                }
                condition_number_idx = test_util.rng_state.choice(len(test_data['event_codes'][0]))
                condition_number = test_data['event_codes'][0][condition_number_idx]

                start_codes = test_data['start_codes']
                end_codes = test_data['end_codes']
                event_splitting_params = {
                    'comment': '',
                    'subtrials': [{'start_code': int(x),
                                   'end_code': int(y)} for (x, y) in zip(start_codes, end_codes)],
                    'trial_to_condition_func': 'lambda code, idx: code[{}]'.format(condition_number_idx),
                    'margin_before': margin_before,
                    'margin_after': margin_after,
                }
                self.assertTrue(validate(EventSplittingParamsSchema.get_schema(), event_splitting_params))

                if not all_codes:
                    event_splitting_params['trial_start_code'] = int(test_data['event_codes'][0][0])
                    event_splitting_params['trial_end_code'] = int(test_data['event_codes'][0][-1])
                    # use single thread to make code coverage detection easier.
                split_result = split_events(event_data, event_splitting_params, n_jobs=1)
                # ok. let's test the result.

                # build correct result
                correct_result = {
                    'event_codes': test_data['event_codes'],
                    'event_times': test_data['event_times_per_trial'],
                    'condition_number': np.full(num_trial, fill_value=condition_number,
                                                dtype=condition_number.dtype),
                    'start_times': test_data['start_times'],
                    'end_times': test_data['end_times'],
                    'trial_start_time': test_data['trial_start_time'],
                    'trial_end_time': test_data['trial_end_time'],
                }
                self.check_result(num_trial, split_result, correct_result)

    def check_result(self, num_trial, split_result, correct_result):
        self.assertEqual((num_trial,), split_result['event_codes'].shape)
        for code1, code2 in zip(correct_result['event_codes'], split_result['event_codes']):
            self.assertTrue(np.array_equal(code1, code2))

        # test event times.
        self.assertEqual((num_trial,), split_result['event_times'].shape)
        for time1, time2 in zip(correct_result['event_times'], split_result['event_times']):
            self.assertEqual(time1.shape, time2.shape)
            self.assertTrue(np.allclose(time1, time2))

        # test condition number
        self.assertTrue(np.array_equal(correct_result['condition_number'],
                                       split_result['condition_number']))
        # test start times
        self.assertEqual(correct_result['start_times'].shape, split_result['start_times'].shape)
        self.assertTrue(np.allclose(correct_result['start_times'], split_result['start_times']))

        # test end times
        self.assertEqual(correct_result['end_times'].shape, split_result['end_times'].shape)
        self.assertTrue(np.allclose(correct_result['end_times'], split_result['end_times']))

        # trial start
        self.assertEqual(correct_result['trial_start_time'].shape, split_result['trial_start_time'].shape)
        self.assertTrue(np.allclose(correct_result['trial_start_time'], split_result['trial_start_time']))

        # trial end
        self.assertEqual(correct_result['trial_end_time'].shape, split_result['trial_end_time'].shape)
        self.assertTrue(np.allclose(correct_result['trial_end_time'], split_result['trial_end_time']))

        # trial length
        correct_trial_length = correct_result['trial_end_time'] - correct_result['trial_start_time']
        self.assertEqual(correct_trial_length.shape, split_result['trial_length'].shape)
        self.assertTrue(np.allclose(correct_trial_length, split_result['trial_length']))


if __name__ == '__main__':
    unittest.main()

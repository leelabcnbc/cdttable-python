import unittest
from cdttable.core.io_event import (_get_times_one_sub_trial, _get_times_subtrials,
                                    _split_events_per_trial, split_events)
from cdttable import test_util
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        test_util.reseed(4)

    def test_split_event_all_codes(self):
        margin_before, margin_after = test_util.generate_margins()
        num_trial_list = [1, 50, None]

        for num_trial_ in num_trial_list:
            test_data = test_util.generate_event_data(margin_before, margin_after, num_trial=num_trial_)
            if num_trial_ is not None:
                num_trial = num_trial_
            else:
                num_trial = len(test_data['event_codes'])
            self.assertEqual(num_trial, len(test_data['event_codes']))
            self.assertEqual(num_trial, len(test_data['event_times_per_trial']))
            self.assertEqual(num_trial, len(test_data['event_times']))
            event_data = {
                'event_codes': test_data['event_codes'],
                'event_times': test_data['event_times']
            }
            condition_number_idx = test_util.rng_state.choice(len(test_data['event_codes'][0]))
            condition_number = test_data['event_codes'][0][condition_number_idx]
            start_code = int(test_data['event_codes'][0][0])
            end_code = int(test_data['event_codes'][0][-1])
            event_splitting_params = {
                'comment': '',
                'subtrials': [
                    {
                        "start_code": start_code,
                        "end_code": end_code,
                    },
                ],
                'trial_to_condition_func': 'lambda code, idx: code[{}]'.format(condition_number_idx),
                'margin_before': margin_before,
                'margin_after': margin_after,
            }
            split_result = split_events(event_data, event_splitting_params, debug=True)
            # ok. let's test the result.
            self.assertEqual((num_trial,), split_result['event_codes'].shape)
            for code1, code2 in zip(test_data['event_codes'], split_result['event_codes']):
                self.assertTrue(np.array_equal(code1, code2))

            # test event times.
            self.assertEqual((num_trial,), split_result['event_times'].shape)
            for time1, time2 in zip(test_data['event_times_per_trial'], split_result['event_times']):
                self.assertEqual(time1.shape, time2.shape)
                self.assertTrue(np.allclose(time1, time2))

            # test condition number
            self.assertTrue(np.array_equal(np.full(num_trial, fill_value=condition_number,
                                                   dtype=condition_number.dtype),
                                           split_result['condition_number']))
            # test start times
            correct_start_times = np.empty((num_trial, 1), dtype=np.float64)
            for i_trial in range(num_trial):
                correct_start_times[i_trial] = test_data['event_times_per_trial'][i_trial][0]
            self.assertEqual(correct_start_times.shape, split_result['start_times'].shape)
            self.assertTrue(np.allclose(correct_start_times, split_result['start_times']))

            # test end times
            correct_end_times = np.empty((num_trial, 1), dtype=np.float64)
            for i_trial in range(num_trial):
                correct_end_times[i_trial] = test_data['event_times_per_trial'][i_trial][-1]
            self.assertEqual(correct_end_times.shape, split_result['end_times'].shape)
            self.assertTrue(np.allclose(correct_end_times, split_result['end_times']))

            # trial start
            self.assertEqual(test_data['trial_start_time'].shape, split_result['trial_start_time'].shape)
            self.assertTrue(np.allclose(test_data['trial_start_time'], split_result['trial_start_time']))

            # trial end
            self.assertEqual(test_data['trial_end_time'].shape, split_result['trial_end_time'].shape)
            self.assertTrue(np.allclose(test_data['trial_end_time'], split_result['trial_end_time']))

            # trial length
            correct_trial_length = test_data['trial_end_time'] - test_data['trial_start_time']
            self.assertEqual(correct_trial_length.shape, split_result['trial_length'].shape)
            self.assertTrue(np.allclose(correct_trial_length, split_result['trial_length']))


if __name__ == '__main__':
    unittest.main()

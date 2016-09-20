import unittest
from cdttable.core.io_event import (split_events, _check_input,
                                    _split_events_per_trial,
                                    _get_times_one_sub_trial,
                                    _get_times_subtrials,
                                    _get_times_trial)
from cdttable import test_util
import numpy as np
from itertools import product
from cdttable.core.schema import validate, EventSplittingParamsSchema


class MyTestCase(unittest.TestCase):
    def setUp(self):
        test_util.reseed(42)

    def test_split_event_code_based(self):
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

    def test_check_input_wrong_dtype(self):
        wrong_time_dtype = []

        wrong_code_dtype = [np.float32, np.uint16, np.uint32, np.uint64, np.int32, np.int64]

        for time_dtype in wrong_time_dtype:
            codes, times = test_util.generate_codes_and_times()
            times = times.astype(time_dtype)
            with self.assertRaises(TypeError):
                _check_input(codes, times)

        for code_type in wrong_code_dtype:
            codes, times = test_util.generate_codes_and_times()
            codes = codes.astype(code_type)
            with self.assertRaises(TypeError):
                _check_input(codes, times)

    def test_check_input_correct_dtype(self):
        correct_time_dtype = [np.float32, np.float64,
                              np.uint8, np.uint16, np.uint32, np.uint64,
                              np.int8, np.int16, np.int32, np.int64]

        correct_code_dtype = [np.uint8, np.int8]

        # there's no assertNotRaises,
        # <http://stackoverflow.com/questions/4319825/python-unittest-opposite-of-assertraises>
        for time_dtype in correct_time_dtype:
            codes, times = test_util.generate_codes_and_times()
            times = times.astype(time_dtype)
            _check_input(codes, times)

        for code_type in correct_code_dtype:
            codes, times = test_util.generate_codes_and_times()
            codes = codes.astype(code_type)
            _check_input(codes, times)

    def test_get_times_one_sub_trial_correct(self):
        time_delay_list = list(test_util.rng_state.rand(10) * 10) + [None]
        for time_delay in time_delay_list:
            with self.subTest(time_delay=time_delay):
                codes, times = test_util.generate_codes_and_times()
                result_ref = test_util.generate_start_and_stop_times(codes, times, num_pair=1, times_delay=[time_delay])
                result_start, result_end = _get_times_one_sub_trial(codes, times, trial_info=result_ref['subtrials'][0])
                self.assertTrue(np.array_equal(result_start, result_ref['start_times'][0]))
                self.assertTrue(np.array_equal(result_end, result_ref['end_times'][0]))

    def test_get_times_one_sub_trial_wrong(self):
        time_delay_list = [1, None]
        for time_delay in time_delay_list:
            with self.subTest(time_delay=time_delay):
                codes, times = test_util.generate_codes_and_times()
                result_ref = test_util.generate_start_and_stop_times(codes, times, num_pair=1, times_delay=[time_delay])
                trial_info = result_ref['subtrials'][0]
                if time_delay is not None:
                    del trial_info['end_time']
                    trial_info['end_Time'] = time_delay
                else:
                    del trial_info['end_code']
                    trial_info['end_Code'] = 1
                with self.assertRaises(RuntimeError):
                    _get_times_one_sub_trial(codes, times, trial_info=trial_info)

    def test_get_times_sub_trials(self):
        for _ in range(100):
            codes, times = test_util.generate_codes_and_times()
            result_ref = test_util.generate_start_and_stop_times(codes, times)
            with self.subTest(times_delay=result_ref['times_delay']):
                result_start, result_end = _get_times_subtrials(codes, times, params=result_ref['subtrials'])
                self.assertTrue(np.array_equal(result_start, result_ref['start_times']))
                self.assertTrue(np.array_equal(result_end, result_ref['end_times']))

    def test_get_times_trial(self):
        trial_type_list = ['simple', 'code', 'time']
        for trial_type in trial_type_list:
            for _ in range(50):
                if test_util.rng_state.rand() > 0.5:
                    margin_before, margin_after = test_util.generate_margins()
                else:
                    margin_before, margin_after = 0, 0
                codes, times = test_util.generate_codes_and_times()
                result_ref = test_util.generate_start_and_stop_times(codes, times)
                params = {'margin_before': float(margin_before), 'margin_after': float(margin_after)}
                trial_start_code = None
                trial_end_code = None
                trial_end_time = None
                if trial_type != 'simple':
                    before_codes = codes[:list(codes).index(result_ref['subtrials'][0]['start_code']) + 1]
                    after_codes = codes[times >= result_ref['end_times'][-1]]
                    if after_codes.size == 0:  # hack. this is impossible in full example.
                        after_codes = [codes[-1]]
                    # choose a start code up to start_codes
                    trial_start_code = test_util.rng_state.choice(before_codes)
                    trial_start_time_abs = times[list(codes).index(trial_start_code)]
                    params['trial_start_code'] = trial_start_code
                    if trial_type == 'code':
                        trial_end_code = test_util.rng_state.choice(after_codes)
                        params['trial_end_code'] = trial_end_code
                    elif trial_type == 'time':
                        # choose a time that is later than end_times
                        trial_end_time_abs = test_util.rng_state.rand() + result_ref['end_times'][-1]
                        trial_end_time = trial_end_time_abs - trial_start_time_abs
                        params['trial_end_time'] = trial_end_time

                result_ref_trial = test_util.generate_trial_start_and_stop_times(codes, times,
                                                                                 result_ref['start_times'],
                                                                                 result_ref['end_times'],
                                                                                 margin_before=margin_before,
                                                                                 margin_after=margin_after,
                                                                                 trial_param_type=trial_type,
                                                                                 start_code=trial_start_code,
                                                                                 end_code=trial_end_code,
                                                                                 end_time=trial_end_time
                                                                                 )
                # generate some start code and stop code my self.
                with self.subTest(trial_type=trial_type, params=params):
                    result_start, result_end = _get_times_trial(codes, times,
                                                                result_ref['start_times'],
                                                                result_ref['end_times'],
                                                                params=params)
                    self.assertTrue(np.allclose(result_start, result_ref_trial[0]))
                    self.assertTrue(np.allclose(result_end, result_ref_trial[1]))

    def test_get_times_trial_wrong(self):
        trial_type_list = ['simple', 'code', 'time']
        for trial_type in trial_type_list:
            for _ in range(50):
                if test_util.rng_state.rand() > 0.5:
                    margin_before, margin_after = test_util.generate_margins()
                else:
                    margin_before, margin_after = 0, 0
                codes, times = test_util.generate_codes_and_times()
                result_ref = test_util.generate_start_and_stop_times(codes, times)
                params = {'margin_before': float(margin_before), 'margin_after': float(margin_after)}
                trial_start_code = None
                trial_end_code = None
                trial_end_time = None
                if trial_type != 'simple':
                    before_codes = codes[:list(codes).index(result_ref['subtrials'][0]['start_code']) + 1]
                    after_codes = codes[times >= result_ref['end_times'][-1]]
                    if after_codes.size == 0:  # hack. this is impossible in full example.
                        after_codes = [codes[-1]]
                    # choose a start code up to start_codes
                    trial_start_code = test_util.rng_state.choice(before_codes)
                    trial_start_time_abs = times[list(codes).index(trial_start_code)]
                    params['trial_start_code'] = trial_start_code
                    if trial_type == 'code':
                        trial_end_code = test_util.rng_state.choice(after_codes)
                        params['trial_end_codE'] = trial_end_code
                    elif trial_type == 'time':
                        # choose a time that is later than end_times
                        trial_end_time_abs = test_util.rng_state.rand() + result_ref['end_times'][-1]
                        trial_end_time = trial_end_time_abs - trial_start_time_abs
                        params['trial_end_timE'] = trial_end_time

                result_ref_trial = test_util.generate_trial_start_and_stop_times(codes, times,
                                                                                 result_ref['start_times'],
                                                                                 result_ref['end_times'],
                                                                                 margin_before=margin_before,
                                                                                 margin_after=margin_after,
                                                                                 trial_param_type=trial_type,
                                                                                 start_code=trial_start_code,
                                                                                 end_code=trial_end_code,
                                                                                 end_time=trial_end_time
                                                                                 )
                # generate some start code and stop code my self.
                with self.subTest(trial_type=trial_type, params=params):
                    if trial_type == 'simple':
                        _get_times_trial(codes, times, result_ref['start_times'], result_ref['end_times'],
                                         params=params)
                    else:
                        with self.assertRaises(RuntimeError):
                            _get_times_trial(codes, times, result_ref['start_times'], result_ref['end_times'],
                                             params=params)


if __name__ == '__main__':
    unittest.main()

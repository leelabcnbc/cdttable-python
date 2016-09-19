import unittest
from cdttable.core.io_event import (_get_times_one_sub_trial, _get_times_subtrials,
                                    _split_events_per_trial, split_events)
from cdttable import test_util
from cdttable.core.schema import _CDTTableImportParamsSchemaCommon, validate


class MyTestCase(unittest.TestCase):
    def setUp(self):
        test_util.reseed(0)

    def test_split_event_all_codes(self):
        margin_before, margin_after = test_util.generate_margins()
        num_trial_list = [1, 50, None]

        for num_trial in num_trial_list:
            test_data = test_util.generate_event_data(margin_before, margin_after, num_trial=num_trial)
            event_data = {
                'event_codes': test_data['event_codes'],
                'event_times': test_data['event_times']
            }
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
                'trial_to_condition_func': 'lambda code, idx: code[0]',
                'margin_before': margin_before,
                'margin_after': margin_after,
            }
            split_result = split_events(event_data, event_splitting_params, debug=True)
            if num_trial == 1:
                print(split_result)
                print(test_data['event_times_per_trial'])



if __name__ == '__main__':
    unittest.main()

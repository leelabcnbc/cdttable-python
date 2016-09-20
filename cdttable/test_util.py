import numpy as np
import numpy.random

rng_state = numpy.random.RandomState(seed=42)


def reseed(seed=None):
    rng_state.seed(seed)


def generate_margins(max_margin=1.0):
    return rng_state.rand(2) * max_margin


def generate_event_data(margin_before=0.0, margin_after=0.0, num_trial=None, num_pair=None, *,
                        max_trial_length=1.0, max_flexible_margin=1.0,
                        all_codes=True  # whether using all event codes, that is, first code being trial start,
                        # and last code being trial end
                        ):
    num_code = rng_state.randint(10, 20 + 1)
    event_codes_per_trial = rng_state.choice(np.arange(-100, 100), size=num_code, replace=False).astype(np.int16)
    if num_trial is None:
        num_trial = rng_state.randint(1, 500 + 1)
    event_codes = [event_codes_per_trial.copy() for _ in range(num_trial)]

    # generate start/end codes.
    if num_pair is None:
        num_pair = rng_state.randint(1, 5 + 1)
    # choose pairs

    code_pair_idx = np.sort(rng_state.choice(num_code, num_pair * 2, replace=False))
    start_code_idx = code_pair_idx[0::2]
    end_code_idx = code_pair_idx[1::2]
    if all_codes:
        num_pair = 1
        start_code_idx = np.array([0])
        end_code_idx = np.array([num_code - 1])
    start_codes = event_codes_per_trial[start_code_idx]
    end_codes = event_codes_per_trial[end_code_idx]
    assert len(start_codes) == len(end_codes) == num_pair

    # then generate some time stamps.
    trial_length_raw = rng_state.rand(num_trial) * max_trial_length

    event_times_truth = [np.concatenate([np.array([0]), np.sort(rng_state.rand(num_code - 1)) * trial_length_raw[i]])
                         for i
                         in range(num_trial)]

    trial_margin_before = rng_state.rand(num_trial) * max_flexible_margin + margin_before
    trial_margin_after = rng_state.rand(num_trial) * max_flexible_margin + margin_after

    trial_length_padded = trial_length_raw + trial_margin_before + trial_margin_after
    trial_length_padded_cum = np.cumsum(trial_length_padded)

    # ok. now let's combine the margins.
    event_times = []
    trial_start_time = []
    trial_end_time = []

    start_times = []
    end_times = []

    for trial_idx in range(num_trial):
        if trial_idx == 0:
            previous_time = 0.0
        else:
            previous_time = trial_length_padded_cum[trial_idx - 1]
        times_truth = event_times_truth[trial_idx]
        times_abs = times_truth + previous_time + trial_margin_before[trial_idx]
        trial_start_time.append(previous_time + trial_margin_before[trial_idx] - margin_before)
        trial_end_time.append(times_abs[-1] + margin_after)

        event_times.append(times_abs)
        event_times_truth[trial_idx] += margin_before

        start_times.append(event_times_truth[trial_idx][start_code_idx])
        end_times.append(event_times_truth[trial_idx][end_code_idx])

    trial_start_time = np.asarray(trial_start_time)
    trial_end_time = np.asarray(trial_end_time)
    start_times = np.asarray(start_times)
    end_times = np.asarray(end_times)

    result = {
        'event_codes': event_codes,
        'event_times': event_times,
        'event_times_per_trial': event_times_truth,
        'trial_start_time': trial_start_time,
        'trial_end_time': trial_end_time,
        'start_times': start_times,
        'end_times': end_times,
        'start_codes': start_codes,
        'end_codes': end_codes
    }

    return result

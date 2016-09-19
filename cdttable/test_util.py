import numpy as np
import numpy.random

rng_state = numpy.random.RandomState(seed=42)


def reseed(seed=None):
    rng_state.seed(seed)


def generate_margins(max_margin=1.0):
    return rng_state.rand(2) * max_margin


def generate_event_data(margin_before=0.0, margin_after=0.0, num_trial=None, *, max_trial_length=1.0,
                        max_flexible_margin=1.0
                        ):
    num_code = rng_state.randint(10, 20 + 1)
    event_codes_per_trial = rng_state.choice(np.arange(-100, 100), size=num_code, replace=False).astype(np.int16)
    if num_trial is None:
        num_trial = rng_state.randint(1, 500 + 1)
    event_codes = [event_codes_per_trial.copy() for _ in range(num_trial)]
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

    trial_start_time = np.asarray(trial_start_time)
    trial_end_time = np.asarray(trial_end_time)

    return {
        'event_codes': event_codes,
        'event_times': event_times,
        'event_times_per_trial': event_times_truth,
        'trial_start_time': trial_start_time,
        'trial_end_time': trial_end_time,
    }

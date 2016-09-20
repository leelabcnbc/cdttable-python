from collections import OrderedDict
from .schema import validate, ImportParamsSchema
from .io_event import split_events
from .io_data import import_one_data_table


def import_sessions(session_gen, import_params: dict):
    """import multiple sesssions

    Parameters
    ----------
    session_gen: an iterable to generate session data one by one
    import_params

    Returns
    -------
    a generator object for getting processed sessions. Thus it can be used in a very
    flexible way. No unnecessary memory consumption.
    """
    return (import_one_session(session, import_params) for session in session_gen)


def import_one_session(session, import_params: dict):
    """

    Parameters
    ----------
    session
    import_params

    Returns
    -------
    some (ordered?) dict object having all columns ready to be saved to hdf5 or mat
    """
    assert validate(ImportParamsSchema.get_schema(), import_params)

    event_splitting_result = split_events(session['event_data'], import_params['event_splitting_params'])
    # first let's work on the time markers events.

    # work out the start and stop times of each trial.
    start_times = None
    stop_times = None

    processed_session_dict = OrderedDict()

    for table_name, table_data_raw in session['data'].items():
        # extract the json describing how this piece of data looks like.
        data_meta_this = import_params['data_meta'][table_name]
        processed_session_dict[table_name] = import_one_data_table(table_data_raw, data_meta_this,
                                                                   event_splitting_result,
                                                                   import_params)


def export_sessions(processed_session_gen, export_format: str, convert_params: dict):
    # let's do this sequentially.
    if export_format == 'mat':
        processor = _export_sessions_mat
    elif export_format == 'hdf5':
        processor = _export_sessions_hdf5
    else:
        raise ValueError('unsupported export format! must be mat or hdf5!')
    processor(processed_session_gen, convert_params)


def _export_sessions_mat():
    pass


def _export_sessions_hdf5():
    pass

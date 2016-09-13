from collections import OrderedDict


def import_sessions(session_gen, import_params):
    """import multiple sesssions

    Parameters
    ----------
    session_gen
    import_params

    Returns
    -------
    a generator object for getting processed sessions. Thus it can be used in a very
    flexible way. No unnecessary memory consumption.
    """
    return (import_one_session(session, import_params) for session in session_gen)


def import_one_session(session, import_params):
    """

    Parameters
    ----------
    session
    import_params

    Returns
    -------
    some (ordered?) dict object having all columns ready to be saved to hdf5 or mat
    """
    # TODO: check import_params is good.

    event_codes = session['event_codes']
    event_times = session['event_times']
    # first let's work on the time markers events.

    processed_session_dict = OrderedDict()

    for table_name, table_data_raw in session['data'].items():
        # extract the json describing how this piece of data looks like.
        data_meta_this = import_params['data_meta'][table_name]
        processor = lambda x: x
        # if data_meta_this['location']:
        #     pass
        #
        # if data_meta_this['continuous']:
        #     pass
        #
        # if data_meta_this['point_process']:
        #     pass
        #
        # if data_meta_this['trial_by_trial']:
        #     pass

        processed_session_dict[table_name] = processor(table_data_raw)


def export_sessions(processed_session_gen, export_format, convert_params):
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

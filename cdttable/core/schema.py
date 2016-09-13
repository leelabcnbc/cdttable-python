import jsonschema
import jsl
import json

_time_aware_data_types = ('time_series_1d',  # LFP
                          'point_process_1d',  # spike
                          )

_time_unaware_fixed = ('fixed',  # a fixed bunch of numbers for each trial/location, with no notion of time
                       )

_time_unaware_flexible = ('variable_1d',  # a variable bunch of numbers for each trial/location, with no notion of time.
                          )

_time_unaware_data_types = sum([_time_unaware_fixed,
                                _time_unaware_flexible],
                               ())
_data_types_all = _time_aware_data_types + _time_unaware_data_types


class DataMetaJSLBase(jsl.Document):
    location_columns = jsl.ArrayField(unique_items=True, required=True)


class DataMetaJSLForTimeAware(DataMetaJSLBase):
    type = jsl.StringField(enum=_time_aware_data_types, required=True)
    split = jsl.BooleanField(required=True)


class DataMetaJSLForTimeUnawareBase(DataMetaJSLBase):
    split = jsl.BooleanField(enum=[True], required=True)  # must be split ahead.


class DataMetaJSLForFixed(DataMetaJSLForTimeUnawareBase):
    type = jsl.StringField(enum=_time_unaware_fixed, required=True)
    shape = jsl.ArrayField(items=jsl.IntField(minimum=1), required=True)  # specify the shape of each trial.
    # if [], means scalar.
    # I support multi dimension here so that for some high d time series, we can simply store them as fixed chunks,
    # instead of a bunch of var len time series to save space.


class DataMetaJSLForFlexible(DataMetaJSLForTimeUnawareBase):
    type = jsl.StringField(enum=_time_unaware_flexible, required=True)


DataMetaJSL = jsl.OneOfField([jsl.DocumentField(DataMetaJSLForTimeAware),
                              jsl.DocumentField(DataMetaJSLForFixed),
                              jsl.DocumentField(DataMetaJSLForFlexible),
                              ], required=True)

if __name__ == '__main__':
    print(json.dumps(DataMetaJSL.get_schema(), indent=2))

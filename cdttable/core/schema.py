import jsonschema
import jsl
from jsonschema import FormatChecker, Draft4Validator
import string

def validate(schema, record):
    jsonschema.validate(instance=record, schema=schema, format_checker=FormatChecker(),
                        cls=Draft4Validator)

    return True


_data_types = ('fixed',
               'variable_1d',
               )

_valid_column_name_pattern = '^[0-9a-z_\\-]+$'
_valid_column_name_chars = string.ascii_lowercase + string.digits + '-_'

_valid_column_name_field = jsl.StringField(pattern=_valid_column_name_pattern, required=True)


class DataMetaJSLBaseNoSplitter(jsl.Document):
    location_columns = jsl.ArrayField(unique_items=True, required=True,
                                      items=_valid_column_name_field)
    value_column = _valid_column_name_field
    type = jsl.StringField(enum=_data_types, required=True)
    additional_parameters = jsl.DictField(required=True)


class DataMetaJSLBaseWithSplitter(jsl.Document):
    # I can't inherit from DataMetaJSLBaseNoSplitter, because some inheritance issue
    # <http://stackoverflow.com/questions/29214888/typeerror-cannot-create-a-consistent-method-resolution-order-mro>
    location_columns = jsl.ArrayField(unique_items=True, required=True,
                                      items=_valid_column_name_field)
    value_column = _valid_column_name_field
    type = jsl.StringField(enum=_data_types, required=True)
    splitter = jsl.StringField(required=True)  # whether we need a special splitter
    splitter_params = jsl.DictField(required=True)
    additional_parameters = jsl.DictField(required=True)


class DataMetaJSLBase(DataMetaJSLBaseNoSplitter, DataMetaJSLBaseWithSplitter):
    class Options:
        inheritance_mode = jsl.ONE_OF


class DataMetaJSLWithTimeStamp(DataMetaJSLBase):
    with_timestamp = jsl.BooleanField(enum=[True], required=True)
    timestamp_type = jsl.StringField(enum=['absolute', 'relative'], required=True)
    timestamp_column = _valid_column_name_field


class DataMetaJSLWithoutTimeStamp(DataMetaJSLBase):
    with_timestamp = jsl.BooleanField(enum=[False], required=True)


# schema for a single data source.
class DataMetaJSL(DataMetaJSLWithTimeStamp, DataMetaJSLWithoutTimeStamp):
    class Options:
        inheritance_mode = jsl.ONE_OF


class _SubTrialEndTime(jsl.Document):
    start_code = jsl.IntField(required=True)
    end_time = jsl.NumberField(minimum=0, required=True)


class _SubTrialEndCode(jsl.Document):
    start_code = jsl.IntField(required=True)
    end_code = jsl.IntField(required=True)


class _SubTrial(_SubTrialEndTime, _SubTrialEndCode):
    class Options:
        inheritance_mode = jsl.ONE_OF


class _CDTTableImportParamsSchemaCommon(jsl.Document):
    comment = jsl.StringField()
    subtrials = jsl.ArrayField(items=jsl.DocumentField(_SubTrial), unique_items=True, required=True, min_items=1)
    margin_before = jsl.NumberField(minimum=0, required=True)  # 0.3 by default in previous implementation.
    margin_after = jsl.NumberField(minimum=0, required=True)  # 0.3 by default in previous implementation.
    trial_to_condition_func = jsl.StringField(required=True)  # should be a function of both event codes and trial idx.


class _CDTTableImportParamsSchemaEndTime(_CDTTableImportParamsSchemaCommon):
    trial_start_code = jsl.IntField(required=True)
    trial_end_time = jsl.NumberField(minimum=0, required=True)


class _CDTTableImportParamsSchemaEndCode(_CDTTableImportParamsSchemaCommon):
    trial_start_code = jsl.IntField(required=True)
    trial_end_code = jsl.IntField(required=True)


# you can also use oneOfField, but that won't give you `$schema` field when using .get_schema().
class EventSplittingParamsSchema(_CDTTableImportParamsSchemaEndTime,
                                 _CDTTableImportParamsSchemaEndCode, _CDTTableImportParamsSchemaCommon):
    class Options:
        inheritance_mode = jsl.ONE_OF


class ImportParamsSchema(jsl.Document):
    """template for the whole import"""
    schema_revision = jsl.IntField(enum=[1], required=True)  # in case of large change later.
    notes = jsl.StringField(required=True)
    data_meta = jsl.ArrayField(items=jsl.DocumentField(DataMetaJSL), unique_items=True, required=True)
    event_splitting_params = jsl.DocumentField(EventSplittingParamsSchema)

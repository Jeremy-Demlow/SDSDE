copy into <DATABASE>.<SCHEMA>.<TABLE_NAME>
from @<STAGE_NAME><DATA_LAKE_PATH>
file_format = (type = <FILE_TYPE> field_delimiter = '<FIELD_DELIMITER>' encoding = '<ENCODING>' compression = <COMPRESSION> skip_header = <SKIP_HEADER>)
pattern = '<PATTERN>';
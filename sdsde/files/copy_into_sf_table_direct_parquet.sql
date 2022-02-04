copy into <DATABASE>.<SCHEMA>.<TABLE_NAME>
from ( <SELECT_STATEMENT>
FROM '<AZURE_PATH>'
)
file_format = (type = <FILE_TYPE> field_delimiter = '<FIELD_DELIMITER>' encoding = '<ENCODING>' compression = <COMPRESSION> skip_header = <SKIP_HEADER>)
credentials= (azure_sas_token = '<AZURE_SAS_TOKEN>')
pattern = '<PATTERN>';
COPY INTO '<AZURE_PATH>'
FROM <TABLE_NAME> (<SF_QUERY>)
partition by = (<PARTITION_BY>)
max_file_size = <MAX_FILE_SIZE>
OVERWRITE = <OVER_WRITE>
file_format = (type = <FILE_TYPE> field_optionally_enclosed_by = '<FIELD_OPTIONALLY_ENCLOSED_BY>' field_delimiter = '<FIELD_DELIMITER>' encoding = '<ENCODING>' compression = <COMPRESSION> skip_header = <SKIP_HEADER>)
credentials= (azure_sas_token = '<AZURE_SAS_TOKEN>')
header = <HEADER>;
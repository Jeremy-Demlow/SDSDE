COPY INTO @<STAGE_NAME>/<DATA_LAKE_PATH>
FROM <TABLE_NAME> (<SF_QUERY>)
partition by = (<PARTITION_BY>)
max_file_size = <MAX_FILE_SIZE>
overwrite = <OVER_WRITE>
header = <HEADER>;
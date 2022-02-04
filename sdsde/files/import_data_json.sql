copy into INSERT_TABLE_NAME_HERE
 from 'azure://INSERT_AZURE_STORAGE_ACCOUNT_NAME_HERE.blob.core.windows.net/INSERT_CONTAINER_NAME_HERE/INSERT_FILE_NAME_HERE'
 credentials=(azure_sas_token='INSERT_AZURE_SAS_TOKEN_HERE')
 encryption=(type= 'NONE')
 file_format = (type = csv field_delimiter = ',' FIELD_OPTIONALLY_ENCLOSED_BY = '"' skip_header = 1)
 on_error = continue;
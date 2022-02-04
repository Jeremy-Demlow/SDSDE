create or replace stage <STAGE_NAME>
url='<URL>'
credentials=(azure_sas_token='<SAS_TOKEN>')
encryption=(type= 'NONE')
file_format = (type = <FILE_TYPE> field_delimiter = '<FIELD_DELIMITER>' encoding = '<ENCODING>' compression = <COMPRESSION> field_optionally_enclosed_by = '<FIELD_OPTIONALLY_ENCLOSED_BY>')
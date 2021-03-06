# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_snowflake_copyinto2.ipynb (unless otherwise specified).

__all__ = ['logger', 'make_data_lake_stage', 'stage_query_generator', 'copy_into_adls_query_generator',
           'copy_into_sf_query_generator', 'parquet_copy_into_sf_query_generator', 'clean_special_chars',
           'create_sf_table_from_df', 'create_sf_table_from_dict', 'return_sf_type']

# Cell
from sdsde import files

import pandas as pd
import sys
import logging
import os


logging.basicConfig(level=logging.INFO)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Cell
def make_data_lake_stage(stage_name: str,
                         account: str,
                         container: str,
                         data_lake_path: str,
                         sas_token: str,
                         file_type: str,
                         compression: str = None,
                         field_delimiter: str = None,
                         field_optionally_enclosed_by: str = None,
                         encoding: str = None):
    """
    creates a data lake staging environment from snowflake this calls ``stage_query_generator``
    which does the manipulation to the sdsde file that has the options currently available be
    sure to rip this whole file out if there is something that you need to add before it can
    be a request add to sdsde.

    how to use:

    ```python
    stage_query = make_data_lake_stage(sf_connection=sf,
                                       stage_name='sdsdestage_test',
                                       account=os.environ['azure_account'],
                                       container='sdsdetesting',
                                       data_lake_path='testing_stage/',
                                       field_delimiter=r",",
                                       compression='None',
                                       encoding='UTF-8',
                                       sas_token=os.environ['DATALAKE_SAS_TOKEN_SECRET'],
                                       file_type='csv'
                                       )
    sf.run_str_query(stage_query)

    ```

    Args:
    * stage_name (str): name of stage in snowflake
    * account (str): blob storage account
    * container (str): blob container
    * data_lake_path (str): path in the container to stage in
    * sas_token (str): shared access token for blob
    * file_type (str): for most use cases csv has been used but parquet and others can be used
    * compression (str): the file type compression None if you want the raw file type like csv
      AUTO | GZIP | BZ2 | BROTLI | ZSTD | DEFLATE | RAW_DEFLATE | NONE
    * encoding (str): file encoding method used to parse files on snowflakes side
    * field_delimiter (str): file type deliminter like ; or /t
    """
    stage_url = f'azure://{account}.blob.core.windows.net/{container}/{data_lake_path}'
    logger.info(f'Datalake Stage path that copy into will use {data_lake_path}')
    stage_query = stage_query_generator(stage_name, stage_url, sas_token, field_delimiter,
                                        encoding, compression, file_type=file_type,
                                        field_optionally_enclosed_by=field_optionally_enclosed_by)
    logger.info(f"stage_query: \n {stage_query.replace(sas_token, '**MASKED**')}")
    return stage_query

# Cell
def stage_query_generator(stage_name: str,
                          url: str,
                          sas_token: str,
                          field_delimiter: str,
                          encoding: str,
                          compression: str,
                          file_type: str,
                          field_optionally_enclosed_by: str):
    """
    generates the snowflake query needed to create an external stage in
    azure blob this is inside of ``make_data_lake_stage`` that is the only
    due to the vars() that makes this a little simpler and more robost for this
    use case.

    TODO: figure out string manipulation inside of a list comp, but is not supported
    in python 3.8 and figure out a better way to have the chained replace calls

    Args:
    * stage_name (str): name of the stage in snowflake
    * url (str): azure formated string for account, container, and path
    * sas_token (str): blob sas token for shared access
    * field_delimiter (str): file type deliminter like ; or /t
    * encoding (str): file encoding method used to parse files on snowflakes side
    * compression (str): the file type compression None if you want the raw file type like csv
    AUTO | GZIP | BZ2 | BROTLI | ZSTD | DEFLATE | RAW_DEFLATE | NONE
    * file_type (str, optional): type of files expected in stage. Defaults to 'parquet'. Can use 'csv' as well.

    Returns:
    * str: snowflake query to create stage
    """
    values = vars()
    with open(os.path.join(files.__path__[0], 'stage_template.sql'), 'r') as f:
        lines = f.read()
        f.close()
    for k, v in values.items():
        if v is not None:
            lines = lines.replace(f'<{k.upper()}>', v)
        else:
            lines = lines.replace(f"'<{k.upper()}>'", '').replace(f"{k} =", '').replace(f"<{k.upper()}>", '')
    return lines

# Cell
def copy_into_adls_query_generator(stage_name: str = None,
                                   azure_sas_token: str = None,
                                   data_lake_path: str = None,
                                   azure_path: str = None,
                                   sf_query: str = None,
                                   table_name: str = None,
                                   field_delimiter: str = None,
                                   partition_by: str = None,
                                   max_file_size: str = None,
                                   header: str = None,
                                   encoding: str = None,
                                   file_type: str = None,
                                   field_optionally_enclosed_by: str = None,
                                   skip_header: str = None,
                                   compression: str = None,
                                   over_write: str = None):
    """
    Generate query to dump snowflake data to an adls stage that has already been created.
    There are a lot of optional arguements to allow the user to have a pleasurable experience
    unlocking more than a user typically needs in our current sdsde stage of technology.

    How To Use:

    Note: that the sf_query could also be sf.execute_file(custom_query) to allow for more complex
    queries to dump to azure for the use case at hand.

    ```python
    sg_query = copy_into_adls_query_generator(stage_name='sdsdestage_test',
                                              azure_sas_token=os.environ['DATALAKE_SAS_TOKEN_SECRET'],
                                              sf_query=r'''SELECT * FROM BIDE_EDWDB_ARA_PROD.dbo.FactScan
                                                       WHERE ECID = 84412913 LIMIT 100''',
                                              data_lake_path='testing_stage/',
                                              max_file_size = '32000',
                                              header='True',
                                              over_write='True')
    sf.run_str_query(sg_query)

    ```
    Args:
    * stage_name (str): name of the stage in snowflake
    * azure_sas_token (str): blob sas token for shared access when used to be able to move to a direct azure location
    * data_lake_path (str, optional): Path inside of the created stage used when stage isn't the direct path
    * azure_path (str, optional): Currently not supported do to not having a storage itergration, but this
    would be a direct azure url path
    * sf_query (str, optional): the sf query to use to dump to adls
    * table_name (str, optional): full table name make sure you add the database.schema unless the sf is initialized
    properly
    * field_delimiter (str, optional): file type deliminter like ; or /t
    * partition_by (str, optional): Currently not tested, but this would allow you to dump file structure to adls in
    a partitioned manner
    * max_file_size (str, optional): this allows snowflake to make files as big as this integer number or as small
    * header (str, optional): True = Give the file columns names | False = Skip header and only dump data
    * encoding (str, optional): file encoding engine to be read. Defaults to None.
    * file_type (str, optional): . Defaults to None.
    * field_optionally_enclosed_by (str, optional): . Defaults to None.
    * skip_header (str, optional): this is to skip rows in a file. Defaults to None.
    * compression (str, optional): what type of compression is the file in. Defaults to None.
    * over_write (str, optional): True = overwrite files that are named the same | False = if file is there fail
    Returns:
    * str: Snowflake Query
    """
    values = vars()
    file_sql = 'copy_into_adls_from_sf_stage.sql' if azure_path is None else 'copy_into_adls_from_sf.sql'
    with open(os.path.join(files.__path__[0], file_sql), 'r') as f:
        lines = f.read()
        f.close()
    lines = lines.replace("type =", '') if file_type is None else lines
    for k, v in values.items():
        if v is not None:
            lines = lines.replace(f'<{k.upper()}>', v)
        else:
            lines = lines.replace(f"<{k.upper()}>", '').replace(f"'<{k.upper()}>'", '').replace(f"{k} =", '').replace(r"''", '')
            lines = lines.replace("partition by = ()", '')
    if azure_path is None:
        logger.info(f'\n{lines}')
    else:
        logger.info(f'\n{lines.replace(azure_sas_token, "**MASKED**")}')
    return lines

# Cell
def copy_into_sf_query_generator(database: str,
                                 schema: str,
                                 table_name: str,
                                 file_type: str,
                                 stage_name: str = None,
                                 data_lake_path: str = None,
                                 azure_path: str = None,
                                 azure_sas_token: str = None,
                                 pattern: str = None,
                                 skip_header: str = None,
                                 compression: str = None,
                                 field_delimiter: str = None,
                                 encoding: str = None):
    """
    Generates query to take data from a snowflake stage and puts the data
    directly into a table requested by this function and only if the table in sf
    doesn't exisit it will fail in the how to will show you how to not have it fail
    out, but the notebook shows how to make a snowflake with the data that is coming
    from adls.

    How to use:

    ```python
    cp_query = copy_into_sf_query_generator(stage_name='sdsdestage_test/',
                                            data_lake_path='testing_stage/',
                                            table_name='sdsde_DELETE_TEST_TABLE',
                                            database=sf.sfDatabase,
                                            schema=sf.sfSchema,
                                            skip_header='1',
                                            field_delimiter=',',
                                            encoding='UTF-8',
                                            file_type='csv',
                                            pattern='.*.csv')
    try:
        sf.run_str_query(cp_query)
    except Exception as e:
        logger.error(f'Error Created Trying to Copy Into sf table {e}')
        logger.warning('Most this table needs to be initialized' )
    ```

    Args:
    * stage_name (str): name of the stage in snowflake
    * data_lake_path (str): Path inside of the created stage used when stage isn't the direct path
    * database (str): snowflake database
    * schema (str): snowflake schema
    * table_name (str): snowflake table name that data will be put into
    * file_type (str): file type that will be ingested
    * pattern (str): either this is grabbing many files or the data_lake_path will be point to one file to ingest
    if there is a patter like .*.csv it will use regex to find files with this regex pattern
    * skip_header (str, optional): during development files with columns failed so skipping the header with column names is needed
    * compression (str, optional): what type of compression is being used for this set of data
    * field_delimiter (str, optional): file type deliminter like ; or /t
    * encoding (str, optional): file encoding method used to parse files on snowflakes side

    Returns:
    * str: snowflake query to create stage
    """
    values = vars()
    file_sql = 'copy_into_sf_table.sql' if azure_path is None else 'copy_into_sf_table_direct.sql'
    with open(os.path.join(files.__path__[0], file_sql), 'r') as f:
        lines = f.read()
        f.close()
    for k, v in values.items():
        if v is not None:
            lines = lines.replace(f'<{k.upper()}>', v)
        else:
            lines = lines.replace(f'<{k.upper()}>', '').replace(f"{k} = ''", '').replace(f"{k} =", '')
    logger.info(f'\n{lines}')
    return lines

# Cell
def parquet_copy_into_sf_query_generator(data_types: dict,
                                         database: str,
                                         schema: str,
                                         table_name: str,
                                         file_type: str,
                                         stage_name: str = None,
                                         data_lake_path: str = None,
                                         azure_path: str = None,
                                         azure_sas_token: str = None,
                                         pattern: str = None,
                                         skip_header: str = None,
                                         compression: str = None,
                                         field_delimiter: str = None,
                                         encoding: str = None,
                                         infer_dtypes: bool = False,
                                         header: bool = True):
    """
    Generates query to take data from a snowflake stage and puts the data
    directly into a table requested by this function and only if the table in sf
    doesn't exisit it will fail in the how to will show you how to not have it fail
    out, but the notebook shows how to make a snowflake with the data that is coming
    from adls.

    How to use:

    ```python
    cp_query = copy_into_sf_query_generator(stage_name='sdsdestage_test/',
                                            data_lake_path='testing_stage/',
                                            table_name='sdsde_DELETE_TEST_TABLE',
                                            database=sf.sfDatabase,
                                            schema=sf.sfSchema,
                                            skip_header='1',
                                            field_delimiter=',',
                                            encoding='UTF-8',
                                            file_type='csv',
                                            pattern='.*.csv')
    try:
        sf.run_str_query(cp_query)
    except Exception as e:
        logger.error(f'Error Created Trying to Copy Into sf table {e}')
        logger.warning('Most this table needs to be initialized' )
    ```

    Args:
    * stage_name (str): name of the stage in snowflake
    * data_lake_path (str): Path inside of the created stage used when stage isn't the direct path
    * database (str): snowflake database
    * schema (str): snowflake schema
    * table_name (str): snowflake table name that data will be put into
    * file_type (str): file type that will be ingested
    * pattern (str): either this is grabbing many files or the data_lake_path will be point to one file to ingest
    if there is a patter like .*.csv it will use regex to find files with this regex pattern
    * skip_header (str, optional): during development files with columns failed so skipping the header with column names is needed
    * compression (str, optional): what type of compression is being used for this set of data
    * field_delimiter (str, optional): file type deliminter like ; or /t
    * encoding (str, optional): file encoding method used to parse files on snowflakes side

    Returns:
    * str: snowflake query to create stage
    """
    values = vars()
    del values['data_types']
    del values['infer_dtypes']
    del values['header']
    file_sql = 'copy_into_sf_table_parquet.sql' if azure_path is None else 'copy_into_sf_table_direct_parquet.sql'
    with open(os.path.join(files.__path__[0], file_sql), 'r') as f:
        lines = f.read()
        f.close()
    for k, v in values.items():
        if v is not None:
            lines = lines.replace(f'<{k.upper()}>', v)
        else:
            lines = lines.replace(f'<{k.upper()}>', '').replace(f"{k} = ''", '').replace(f"{k} =", '')
    query = """SELECT FEATURES_HERE"""
    ind = 0
    columns = len(data_types.keys())
    if header is True:
        for k, v in data_types.items():
            query = query.replace('FEATURES_HERE', f'$1:"{k}", FEATURES_HERE')
            if ind < columns:
                query = query.replace(', FEATURES_HERE \n', '')
            else:
                query = query.replace('FEATURES_HERE \n', '')
            ind += 1
    else:
        for k, v in data_types.items():
            query = query.replace('FEATURES_HERE', f'$1:"_COL_{ind}::{return_sf_type(str(v), varchar=False, infer=infer_dtypes)}" as {k}, FEATURES_HERE')
            if ind < columns:
                query = query.replace(', FEATURES_HERE \n', '')
            else:
                query = query.replace('FEATURES_HERE \n', '')
            ind += 1
    query = query.replace(', FEATURES_HERE', '')
    lines = lines.replace('<SELECT_STATEMENT>', query)
    if azure_path is None:
        logger.info(f'\n{lines}')
    else:
        logger.info(f'\n{lines.replace(azure_sas_token, "**MASKED**")}')
    return lines

# Cell
def clean_special_chars(text):
    """
    small nlp clean up tool to take odd characters that could be
    in vendor data inside of column names and then replaces empty
    spaces with ``_``

    Args:
        text (str): dataframe column names as strings

    Returns:
        str: clean column name
    """
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""?????????' + '?????????????????????????????????????????\????????????????&'  # noqa:
    punct += '??^??` <?????????????? ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'  # noqa:
    for p in punct:
        text = text.replace(p, ' ')
        text = text.replace(' ', '_')
    return text

def create_sf_table_from_df(df: pd.DataFrame, table_name_sf: str, varchar: bool):
    """
    Dynamically create a table from a dataframe and
    change the dtypes to snowflake dytpes this may have
    a limitation, but can be added.

    Args:
    * df (pd.DataFrame): data frame to get dtypes
    * table_name_sf (str): snowflake table name
    * varchar: (bool, optional): this will default all dytpes to varchars if True.
    """
    select_query = f'''
        create or replace table {table_name_sf} (FEATURES_HERE);
        '''
    for k, v in dict(df.dtypes).items():
        select_query = select_query.replace('FEATURES_HERE', f'{clean_special_chars(k)} {return_sf_type(str(v), varchar=varchar)}, FEATURES_HERE')
    select_query = select_query.replace(', FEATURES_HERE', '')
    logger.info(select_query)
    logger.warning('Note: Remember this is created data types for sf based on this file if')
    return select_query

def create_sf_table_from_dict(columns_and_types: dict, table_name_sf: str, varchar: bool, infer_types: bool = False):
    """
    Dynamically create a table from a dataframe and
    change the dtypes to snowflake dytpes this may have
    a limitation, but can be added.

    Args:
    * df (pd.DataFrame): data frame to get dtypes
    * table_name_sf (str): snowflake table name
    * varchar: (bool, optional): this will default all dytpes to varchars if True.
    """
    select_query = f'''
        create or replace table {table_name_sf} (FEATURES_HERE);
        '''
    for k, v in columns_and_types.items():
        select_query = select_query.replace('FEATURES_HERE', f'{clean_special_chars(k)} {return_sf_type(str(v), infer=infer_types, varchar=varchar)}, FEATURES_HERE')
    select_query = select_query.replace(', FEATURES_HERE', '')
    logger.info(select_query)
    logger.warning('Note: Remember this is created data types for sf based on this file if')
    return select_query

def return_sf_type(dtype: str, varchar: bool, infer: bool = True):
    """
    Simple utility function that tries to make the process of making a
    snowflake table dynamic and there are of course situtation this will fail
    this is trying to solve 80% of all the data types

    TODO: make more robust if possible, but for now complications will lead
    to the user needing to just default everything to a VARCHAR search bool
    types that didn't work at the time of creation of this function

    Args:
    * dtype (str): dtype from a df in sting form
    * varchar (bool): to default all variables to VARCHAR

    Returns:
    * str: snowflake dtype
    """
    if infer is True:
        if varchar is True:
            dtype = 'VARCHAR'
        elif 'int' in dtype.lower():
            dtype = 'NUMBER'
        elif 'float' in dtype.lower():
            dtype = 'FLOAT'
        elif 'object' in dtype.lower():
            dtype = 'VARCHAR'
        elif 'bool' in dtype.lower():
            dtype = 'VARCHAR'  # TODO: Limitation found before change once resloved by sf
        elif 'date' in dtype.lower():
            dtype = 'DATETIME'  # TODO: Might break with certain datetimes most generic
        else:
            logger.error('odd dtype not seen needs to be resloved...')
            sys.exit()
    else:
        return dtype
    return dtype
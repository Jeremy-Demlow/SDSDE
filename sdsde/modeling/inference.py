# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_modeling_inference.ipynb (unless otherwise specified).

__all__ = ['logger', 'pull_sklearn_object_from_data_lake', 'push_dataframe_to_data_lake_as_parquet',
           'move_parquet_table_to_snowflake', 'query_and_push_feature_set_to_data_lake']

# Cell
import os
import pickle
import pyarrow
import shutil
import pyarrow.parquet as pq
import logging
import numpy as np

from ..wrapper.azurewrapper import blob_pusher, blob_puller
from .premodel import make_data_lake_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell


def pull_sklearn_object_from_data_lake(file_name: str, path: str, container: str, connection_str: str):
    """pulls a pickeld sklearn object from azure data lake to memory

    Args:
    * file_name (str): name of file
    * path (str): data lake path
    * container (str): data lake container
    * connection_str (str): azure connection string for the account

    Returns:
    * (sklearn object): sklearn object loaded from azure
    """
    logger.info(f'Loading Sklearn Object: {os.path.join(path, file_name)}')
    blob_puller(files=[os.path.join(path, file_name)],
                connection_str=connection_str,
                container_name=container,
                drop_location='.',
                overwrite=True)
    with open(file_name, 'rb') as f:
        pipeline = pickle.load(f)
    os.unlink(file_name)
    logger.info('Sklearn Object Loaded')
    return pipeline

# Cell


def push_dataframe_to_data_lake_as_parquet(df, path, container, connection_str,
                                           partition_cols: list = ["partitionidx"], overwrite=True):
    """takes a pandas dataframe and writes it to azure via pyarrow with parquet files

    Args:
    * df (pd.DataFame): dataframe
    * path (str): data lake path
    * container (str): data lake container
    * connection_str (str): azure connection string
    * partition_cols (list, optional): how to partition. fake partitions for speed make on default. Defaults to ["partitionidx"].
    * overwrite (bool, optional): do you overwrite what is there now. Defaults to True.
    """

    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info(f'Removing existing files to write a new batch from {path}')

    if partition_cols[0] == "partitionidx":
        n_partition = int(np.ceil(df.shape[0] / 50000))
        df["partitionidx"] = np.random.choice(range(n_partition), size=df.shape[0])
        logger.info(f'Partitioning column created for distribution with {n_partition} partitions')

    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(table, root_path=path, partition_cols=partition_cols)
    logger.info('Parquet file staged in local disk memory')

    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.parquet']
    for file_name in all_files:
        logger.info(f'Moving File: {file_name}')
        blob_pusher(container_name=container,
                    connection_str=connection_str,
                    file_path=[file_name],
                    blob_dest=[os.path.dirname(file_name)],
                    overwrite=overwrite)

    shutil.rmtree(path)
    logger.info('Local parquet files removed')

# Cell


def move_parquet_table_to_snowflake(sf_connection, table_name: str, stage_name: str,
                                    path: dict, columns_and_types: dict,
                                    pattern: str, replace_table: bool = True):
    """moves data sitting in a parquet format in ADLS to a snowflake table

    Args:
    * sf_connection (SnowflakeConnect): snowflake connection
    * table_name (str): table name
    * stage_name (str): snowflake stage name
    * path (str): path in ADLS to parquet data
    * columns_and_types (dict): snowflake column namees and types
    * pattern (str): pattern for reading files from ADLS
    * replace_table (bool, optional): true does create or relace, false does insert. Defaults to True.
    """

    if replace_table is False:
        select_query = f'''
        insert into {table_name}
            select
                FEATURES_HERE
            from @{stage_name + path} (pattern=>'{pattern}')
        '''
    else:
        select_query = f'''
        create or replace table {table_name} as
            select
                FEATURES_HERE
            from @{stage_name + path} (pattern=>'{pattern}')
        '''
    for k, v in columns_and_types.items():
        select_query = select_query.replace('FEATURES_HERE', f'$1:"{k}"::{v.upper()} as {k}, FEATURES_HERE')
    select_query = select_query.replace(', FEATURES_HERE', '')
    logger.info(select_query)
    sf_connection.run_str_query(select_query)

# Cell


def query_and_push_feature_set_to_data_lake(sf_connection: object, query_file_path: str,
                                            stage_name: str, account: str,
                                            container: str, data_lake_path: str,
                                            blob_path: str, sas_token: str,
                                            connection_str: str, overwrite=True):
    """
    Take a in RAM data set and parition out data set into parquet files
    that are then sent to Azure Data Lake. This assumes that the feature
    store isn't being used for this project. The use case for this is
    to save training/test and predictions to azure that will then be
    sent to snowflake.

    Args:
    * sf_connection (SnowFlake Engine): Snowflake connection
    * query_file_path (str): Path to file to execute
    * stage_name (str): Stage Name for snowflake
    * account (str): Azure blob account name
    * container (str): Container in blob account
    * data_lake_path (str): root level for stage name allowing for re-use
    * blob_path (str): path in container to store data
    * sas_token (str): SAS token found in Access Keys
    * connection_str (str): connection str to azure blob found in Access Keys in Azure
    * overwrite (bool, optional): Overwrite files. Defaults to True.
    """
    logger.info('creating datalake staging area')
    make_data_lake_stage(sf_connection=sf_connection,
                         stage_name=stage_name,
                         account=account,
                         container=container,
                         data_lake_path=data_lake_path,
                         sas_token=sas_token)
    logger.info('begin query....')
    df = sf_connection.execute_file(query_file_path)
    df.columns = [x.lower() for x in df.columns]
    logger.info(f'query complete files being written to {data_lake_path}')
    push_dataframe_to_data_lake_as_parquet(df=df,
                                           container=container,
                                           path=blob_path,
                                           connection_str=connection_str,
                                           overwrite=overwrite)
# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_modeling_inference_fastai.ipynb (unless otherwise specified).

__all__ = ['logger', 'pull_fastai_learner_from_data_lake', 'pull_fastai_preprocess_from_data_lake',
           'pull_transform_predict_sklearn', 'push_prediction_to_dl_and_sf']

# Cell
import os
import logging
import pandas as pd

from ..wrapper.azurewrapper import blob_puller
from .inference import pull_sklearn_object_from_data_lake, push_dataframe_to_data_lake_as_parquet, move_parquet_table_to_snowflake
from .preprocessingfastai import load_pandas
from fastai.learner import load_learner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell


def pull_fastai_learner_from_data_lake(file_name: str, path: str, container: str,
                                       connection_str: str, overwrite: bool = True,
                                       cpu: bool = True):
    """
    Pulling save fastai tabular model from azure blob storage.

    Args:
    * file_name (str): Model Name/ File name
    * path (str): Path location in azure blob and will be saved in the same location locally
    * container (str): Container model is in
    * connection_str (str): Connection String to Azure Storage
    * overwrite (bool, optional): Overwrite model if locally exists. Defaults to True.
    * cpu (bool, optional): CPU or False For GPU inference. Defaults to True.

    Returns:
    * Tabular Learner: Model
    """
    logger.info(f'loading learner object: {os.path.join(path, file_name)}')
    blob_puller(files=[os.path.join(path, file_name)],
                connection_str=connection_str,
                container_name=container,
                drop_location=path,
                overwrite=overwrite)
    learner = load_learner(os.path.join(path, file_name), cpu=cpu)
    os.unlink(os.path.join(path, file_name))
    logger.info('learner object loaded')
    return learner

# Cell


def pull_fastai_preprocess_from_data_lake(file_name: str, path: str, container: str,
                                          connection_str: str, overwrite: bool = True):
    """
    Pull preprocess object to extraploate process onto a new training set.

    ```
    example:
    dl_test = transformer.train.new(df_test)
    dl_test.process()
    X_test = dl_test.xs
    y_test = dl_test.y
    ```
    Args:
    * file_name (str): file name
    * path (str): Path location in azure blob and will be saved in the same location locally
    * container (str): Container model is in
    * connection_str (str): Connection String to Azure Storage
    * overwrite (bool, optional): Overwrite preprocess object if locally exists. Defaults to True.

    Returns:
    * Tranformer: transformer to prepare new data set for model ingestion
    """
    logger.info(f'loading preprocess object: {os.path.join(path, file_name)}')
    blob_puller(files=[os.path.join(path, file_name)],
                connection_str=connection_str,
                container_name=container,
                drop_location=path,
                overwrite=overwrite)
    transformer = load_pandas(os.path.join(path, file_name))
    logger.info('preprocess object loaded')
    return transformer

# Cell


def pull_transform_predict_sklearn(df, snowflake_connection, model_file_name: str,
                                   model_file_path: str, container: str, connection_str: str,
                                   transformer_path: str, transformer_name: str,
                                   overwrite: bool = True, save_model: bool = True, model=None):
    """
    predict on test set and send those predictions to azure data lake.

    Args:
    * model (sklearn model): Model Classifier
    * snowflake_connection (sdsde function): Creation snowflake engine
    * model_file_name (str): file name
    * model_file_path (str): blob & local storage
    * container (str): container name
    * connection_str (str): Azure blob connection
    * transformer_path (str): Blob location of tranformer for preprocessing
    * transformer_name (str): name of transformer
    * test_query (str): query to query for test
    * overwrite (bool, optional): overwrite results. Defaults to True.

    Returns:
    * list: Test set, probabilities, predictions, model and transformer
    """
    transformer = pull_fastai_preprocess_from_data_lake(file_name=transformer_name,
                                                        path=transformer_path,
                                                        container=container,
                                                        connection_str=connection_str,
                                                        overwrite=overwrite)
    model = pull_sklearn_object_from_data_lake(file_name=model_file_name,
                                               path=model_file_path,
                                               container=container,
                                               connection_str=connection_str)

    dl_test = transformer.train.new(df)
    dl_test.process()
    X_test = dl_test.xs
    y_test = dl_test.y
    assert X_test.shape[0] == y_test.shape[0], 'y_test and x_test have different number of rows'
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    return df, probs, preds, model, transformer

# Cell


def push_prediction_to_dl_and_sf(prediction_df: pd.DataFrame, snowflake_connection, df_col_types: dict,
                                 prediction_path: str, sf_table_name: str, stage_name: str, stage_path: str,
                                 pattern: str, replace_table: bool, container: str, connection_str: str,
                                 overwrite: bool = True):
    """
    A wrapper on a few sdsde functions that will push prediction in memory data set
    to azure data lake and snowflake table.

    Args:
    * prediction_df (pd.DataFrame): Data Frame
    * snowflake_connection (sdsde_function): Snowflake engine connection
    * df_col_types (dict): col names and snowflake data types
    * prediction_path (str): store predictions path
    * sf_table_name (str): snowflake table name for predictions
    * stage_name (str): Azure Data Lake Stage name
    * stage_path (str): Stage Path
    * pattern (str): pattern to read paritions
    * replace_table (bool): True creates a new table False inserts to exisiting table
    * container (str): container name
    * connectin_str (str): Azure connection str
    * overwrite (bool, optional): overwrite files. Defaults to True.
    """
    push_dataframe_to_data_lake_as_parquet(prediction_df,
                                           path=prediction_path,
                                           container=container,
                                           connection_str=connection_str,
                                           overwrite=overwrite)
    move_parquet_table_to_snowflake(sf_connection=snowflake_connection,
                                    table_name=sf_table_name,
                                    stage_name=stage_name,
                                    path=stage_path,
                                    columns_and_types=df_col_types,
                                    pattern=pattern,
                                    replace_table=replace_table)
    logger.info(f'Preview {sf_table_name} {snowflake_connection.run_str_query(f"SELECT * FROM {sf_table_name} LIMIT 10;")}')
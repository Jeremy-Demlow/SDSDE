# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_modeling_preprocessing.ipynb (unless otherwise specified).

__all__ = ['logger', 'get_cont_cols', 'get_cat_cols', 'generate_sklearn_preprocessing_pipeline']

# Cell
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

import sklearn.preprocessing as sklearnpre
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell


def get_cont_cols(df, cols):
    """
    helper function for making pipelines
    """
    return df[cols]


def get_cat_cols(df, cols):
    """
    helper function for making pipelines
    """
    return df[cols].astype(str)

# Cell


def generate_sklearn_preprocessing_pipeline(feature_dict, impute=True, impute_strategy='mean'):
    """Given a correctly formated feature dictionary, this function will create
    (without fitting) a sklearn pipeline for preprocessing. The
    function accepts a list of feature keys and transformer values. The
    specified transformers should be from the `sklearn.preprocessing` module,
    hence the name of the function. Arguments to the tranformer can also be passed
    in with the dictionary. Imputing boolean and strategy are also accepted.
    In yaml format, here would be an acceptable feature dictionary definition.

    ```
    MARKETINGZONE:
        variable_type:cont
        transformation:
            name: OrdinalEncoder
            args: {}
    ONLYSINGLERESORTKEY:
        variable_type:cont
        transformation:
            name:OneHotEncoder
            args:
                handle_unknown: ignore
    TOTALSEASONSSCANNED:
        variable_type:cont
        transformation:
            name:StandardScaler
            args: {}
    MAXSEASONVISITATIONSTREAK:
        variable_type:cont
        transformation:
            name:RobustScaler
            args: {}
    ```

    Args:
    * feature_dict (dict): definition for feature transformations
    * impute (bool): impute values at the end or not
    * impute_strategy (str): how to impute values. default is mean

    Returns:
    * object: sklearn feature union pipeline
    """
    logger.info('Creating Sklearn Preprocessing Pipeline')
    pipeline = []
    for feature in feature_dict:
        transformer = getattr(sklearnpre, feature_dict[feature]['transformation']['name'])(**feature_dict[feature]['transformation']['args'])
        logger.info(f'Feature: {feature} --> Transformer: {transformer}')
        if feature_dict[feature]['variable_type'] == 'cont':
            pipeline.append(make_pipeline(FunctionTransformer(get_cont_cols, kw_args={'cols': [feature]}, validate=False), transformer))
        else:
            pipeline.append(make_pipeline(FunctionTransformer(get_cat_cols, kw_args={'cols': [feature]}, validate=False), transformer))
    preprocess_pipeline = make_union(*pipeline)
    if impute:
        logger.info(f'Imputing missing data with {impute_strategy} strategy')
        preprocess_pipeline = Pipeline([
            ('preprocessing', preprocess_pipeline),
            ('imputing', SimpleImputer(strategy=impute_strategy))
        ])
    else:
        logger.info('No imputing for this pipeline')
    logger.info(f'Preprocessing Pipeline Object:\n{preprocess_pipeline}')
    return preprocess_pipeline
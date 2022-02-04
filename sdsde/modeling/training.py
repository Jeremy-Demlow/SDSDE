# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_modeling_training.ipynb (unless otherwise specified).

__all__ = ['logger', 'HpOptMultilabel', 'HpOptBinary', 'HpOptRegression', 'HpOptFeatureSelection',
           'save_sklearn_object_to_data_lake']

# Cell
import xgboost
import os
import pickle
import logging

from hyperopt import fmin, tpe, STATUS_OK, Trials, hp, space_eval
from ..wrapper.azurewrapper import blob_pusher
from sklearn import metrics
from fastai.basics import *
from fastai.tabular.all import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell


class HpOptMultilabel:
    """Class that hypertunes an arbitrary model to multilabel classification
    """

    def __init__(self, X_train, X_test, y_train, y_test, parameter_space=None, model=xgboost.XGBClassifier):
        """Initialization takes in a test and train set and an optional hyperparameter space

        Args:
        * X_train (array): training features
        * X_test (array): testing features
        * y_train (array): testing labels
        * y_test (array): testing labels
        * parameter_space (dict): hyperopt compatible parameter space
        * model (module pointer): machine learning model compatiable with parameter space
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

        if parameter_space is None:
            self.parameter_space = {
                'max_depth': hp.choice('max_depth', np.arange(21, dtype=int) + 2),
                'reg_alpha': hp.uniform('reg_alpha', 0, 5),
                'reg_lambda': hp.uniform('reg_lambda', 0, 5),
                'min_child_weight': hp.uniform('min_child_weight', 0, 5),
                'gamma': hp.uniform('gamma', 0, 5),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'tree_method': hp.choice('tree_method', ['hist', 'exact', 'approx']),
                'objective': hp.choice('objective', ['multi:softmax', 'multi:softprob']),
                'eval_metric': hp.choice('eval_metric', ['mlogloss', 'merror']),
                'gpu_id': hp.choice('gpu_id', [0]),
                'use_label_encoder': hp.choice('use_label_encoder', [False]),
            }
        else:
            self.parameter_space = parameter_space

    def objective(self, params):
        """Objective function for loss that is provided to perform the MINLP
        optimizaiton in hyperopt

        Args:
        * params (dict): hyperopt formated dictionary of hyperparameters

        Returns:
        * dict: loss and status for hyperopt optimization
        """
        model = self.model(**params)
        model.fit(self.X_train, self.y_train)
        pred_proba = model.predict_proba(self.X_test)
        loss = 1 - metrics.roc_auc_score(self.y_test, pred_proba, multi_class='ovr', average='macro')
        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self, max_evals=20):
        """optimizes the hyperparameter space in the object

        Args:
        * max_evals: number of hyperopt iterations

        Returns:
        * dict: best hyperparameters
        """
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.parameter_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        return space_eval(self.parameter_space, best)

# Cell


class HpOptBinary:
    """Class that hypertunes an arbitrary model to binary classification
    """

    def __init__(self, X_train, X_test, y_train, y_test, parameter_space=None, model=xgboost.XGBClassifier):
        """Initialization takes in a test and train set and an optional hyperparameter space

        Args:
        * X_train (array): training features
        * X_test (array): testing features
        * y_train (array): testing labels
        * y_test (array): testing labels
        * parameter_space (dict): hyperopt compatible parameter space
        * model (module pointer): machine learning model compatiable with parameter space
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

        if parameter_space is None:
            self.parameter_space = {
                'max_depth': hp.choice('max_depth', np.arange(21, dtype=int) + 2),
                'reg_alpha': hp.uniform('reg_alpha', 0, 5),
                'reg_lambda': hp.uniform('reg_lambda', 0, 5),
                'min_child_weight': hp.uniform('min_child_weight', 0, 5),
                'gamma': hp.uniform('gamma', 0, 5),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'tree_method': hp.choice('tree_method', ['hist', 'exact', 'approx']),
                'objective': hp.choice('objective', ['binary:logistic', 'binary:logitraw', 'binary:hinge']),
                'eval_metric': hp.choice('eval_metric', ['logloss', 'error', 'auc', 'aucpr', 'map']),
                'gpu_id': hp.choice('gpu_id', [0]),
                'use_label_encoder': hp.choice('use_label_encoder', [False]),
            }
        else:
            self.parameter_space = parameter_space

    def objective(self, params):
        """Objective function for loss that is provided to perform the MINLP
        optimizaiton in hyperopt

        Args:
        * params (dict): hyperopt formated dictionary of hyperparameters

        Returns:
        * dict: loss and status for hyperopt optimization
        """
        model = self.model(**params)
        model.fit(self.X_train, self.y_train)
        pred_proba = model.predict_proba(self.X_test)
        loss = 1 - metrics.roc_auc_score(self.y_test, pred_proba[:, 1])
        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self, max_evals=20):
        """optimizes the hyperparameter space in the object

        Args:
        * max_evals: number of hyperopt iterations

        Returns:
        * dict: best hyperparameters
        """
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.parameter_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        return space_eval(self.parameter_space, best)

# Cell


class HpOptRegression:
    """Class that hypertunes an arbitrary model to regression classification
    """

    def __init__(self, X_train, X_test, y_train, y_test, parameter_space=None, model=xgboost.XGBRegressor):
        """Initialization takes in a test and train set and an optional hyperparameter space

        Args:
        * X_train (array): training features
        * X_test (array): testing features
        * y_train (array): testing labels
        * y_test (array): testing labels
        * parameter_space (dict): hyperopt compatible parameter space
        * model (module pointer): machine learning model compatiable with parameter space
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

        if parameter_space is None:
            self.parameter_space = {
                'max_depth': hp.choice('max_depth', np.arange(21, dtype=int) + 2),
                'reg_alpha': hp.uniform('reg_alpha', 0, 5),
                'reg_lambda': hp.uniform('reg_lambda', 0, 5),
                'min_child_weight': hp.uniform('min_child_weight', 0, 5),
                'gamma': hp.uniform('gamma', 0, 5),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'tree_method': hp.choice('tree_method', ['hist', 'exact', 'approx']),
                'objective': hp.choice('objective', ['reg:squarederror', 'reg:squaredlogerror']),
                'eval_metric': hp.choice('eval_metric', ['rmse', 'mae', 'mape', 'rmsle']),
                'gpu_id': hp.choice('gpu_id', [0]),
            }
        else:
            self.parameter_space = parameter_space

    def objective(self, params):
        """Objective function for loss that is provided to perform the MINLP
        optimizaiton in hyperopt

        Args:
        * params (dict): hyperopt formated dictionary of hyperparameters

        Returns:
        * dict: loss and status for hyperopt optimization
        """
        model = self.model(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        loss = metrics.mean_squared_error(self.y_test, y_pred)
        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self, max_evals=20):
        """optimizes the hyperparameter space in the object

        Args:
        * max_evals: number of hyperopt iterations

        Returns:
        * dict: best hyperparameters
        """
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.parameter_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        return space_eval(self.parameter_space, best)

# Cell


class HpOptFeatureSelection:
    """Uses hyperopt to remove features while maximizing an objective for a given problem
    """
    def __init__(self, X_train, X_test, y_train, y_test, space, model, problem_type):
        """Initialize data, model, and problem type

        Args:
        * X_train (DataFrame): training dataframe of features
        * X_test (DataFrame): testing dataframe of labels
        * y_train (DataFrame): training dataframe of features
        * y_test (DataFrame): testing dataframe of labels
        * space (dict): dictionary with each feature corresponding to a hyperopt choice object
        * model (object): model with "fit" and "predict" functions that are callable
        * problem_type (str): one of "binary", "regression", or "multilabel"
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.parameter_space = space
        self.model = model
        self.problem_type = problem_type

    def objective_binary(self, params):
        """binary loss objective that grabs the columns for feature selection. uses AUC metric

        Args:
        * params (dict): which columns to use as features

        Returns:
        * dict: loss and status for hyperopt optimization
        """
        cols = [i for i, j in params.items() if j == 1]
        self.model.fit(self.X_train[cols], self.y_train)
        pred_proba = self.model.predict_proba(self.X_test[cols])
        loss = 1 - metrics.roc_auc_score(self.y_test, pred_proba[:, 1])
        return {'loss': loss, 'status': STATUS_OK}

    def objective_multi(self, params):
        """multilabel loss objective that grabs the columns for feature selection. uses AUC metric

        Args:
        * params (dict): which columns to use as features

        Returns:
        * dict: loss and status for hyperopt optimization
        """
        model = self.model(**params)
        model.fit(self.X_train, self.y_train)
        pred_proba = model.predict_proba(self.X_test)
        loss = 1 - metrics.roc_auc_score(self.y_test, pred_proba, multi_class='ovr', average='macro')
        return {'loss': loss, 'status': STATUS_OK}

    def objective_regression(self, params):
        """regression loss objective that grabs the columns for feature selection. uses MSE metric

        Args:
        * params (dict): which columns to use as features

        Returns:
        * dict: loss and status for hyperopt optimization
        """
        model = self.model(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        loss = metrics.mean_squared_error(self.y_test, y_pred)
        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self, max_evals=20):
        """optimizes a feature space for each type of problem

        Args:
        * max_evals (int, optional): number of hyperopt evaluations. Defaults to 20.

        Returns:
        * object: hyperopt optimized object of parameters which are features
        """
        trials = Trials()
        if self.problem_type == 'binary':
            best = fmin(fn=self.objective_binary,
                        space=self.parameter_space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials)
        elif self.problem_type == 'multilabel':
            best = fmin(fn=self.objective_multi,
                        space=self.parameter_space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials)
        elif self.problem_type == 'regression':
            best = fmin(fn=self.objective_regression,
                        space=self.parameter_space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials)
        else:
            logger.info('Not an acceptable problem type to solve')
            return None
        return space_eval(self.parameter_space, best)

# Cell


def save_sklearn_object_to_data_lake(model, file_name, path, container, connection_str, overwrite=False):
    """moves a sklearn object to azure data lake as a pickle file at a given path

    Args:
    * model (sklearn object): model, pipeline, transformer in sklearn format
    * file_name (str): name of file
    * path (str): data lake path
    * container (str): data lake container
    * connection_str (str): azure connection string for the account
    * overwrite (bool, optional): set to overwrite a current file if there`. Defaults to False.
    """
    logger.info(f'Pushing Sklearn Object to Azure: {os.path.join(path, file_name)}')
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    blob_pusher(container_name=container,
                connection_str=connection_str,
                file_path=[file_name],
                blob_dest=[path],
                overwrite=overwrite)
    os.unlink(file_name)
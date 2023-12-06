# File: XGBPredictor.py
# Description:
#   Definition of the XGBPredictor class used for predicting performance in NAS problems using XGBoost.
#   XGBPredictor (inherits from the Predictor class in predictor.py) and handles tasks such as
#   saving/loading model states, fitting the model, making predictions, and setting hyperparameters.

# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0

import os
import pickle

import numpy as np
import xgboost as xgb
from scipy.stats import loguniform

from .predictor import Predictor


class XGBPredictor(Predictor):

    def __init__(self, ss_type='nasbench101', encoding_type='adj_one_hot'):
        super().__init__(ss_type=ss_type, encoding_type=encoding_type)
        self.model = None
        self.mean = None
        self.std = None
        self.hyperparams = None
        self.default_hyperparams = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 6,
            'min_child_weight': 1,
            'colsample_bytree': 1,
            'learning_rate': 0.3,
            'colsample_bylevel': 1,
        }

    def save(self, file_path):
        """Save the trained model to a file"""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        data_to_save = {
            'model': self.model,
            'mean': self.mean,
            'std': self.std,
            'hyperparams': self.hyperparams
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load(self, file_path):
        """Load a trained model from a file"""
        with open(file_path, 'rb') as f:
            data_loaded = pickle.load(f)

        self.model = data_loaded['model']
        self.mean = data_loaded['mean']
        self.std = data_loaded['std']
        self.hyperparams = data_loaded['hyperparams']

    def fit(self, xtrain, ytrain):
        """Fit the model to the training data"""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        if self.model is None:
            self.model = xgb.XGBRegressor(**self.hyperparams)

        # Normalize the data
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain = (ytrain - self.mean) / self.std

        self.model.fit(xtrain, ytrain)
        train_pred = self.model.predict(xtrain)
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error

    def refit(self, new_archs, new_accs):
        """Refit the model to new training data"""
        self.fit(new_archs, new_accs)

    def predict(self, xtest):
        """Query the model for predictions on the test data"""
        if self.model is None:
            raise RuntimeError('The model has not been fitted yet.')
        predictions = self.model.predict(xtest)
        # Denormalize the predictions
        return predictions * self.std + self.mean

    def set_random_hyperparams(self):
        """Set random hyperparameters for the model"""
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()
        else:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'max_depth': int(np.random.choice(range(1, 15))),
                'min_child_weight': int(np.random.choice(range(1, 10))),
                'colsample_bytree': np.random.uniform(0.0, 1.0),
                'learning_rate': loguniform(0.001, 0.5).rvs(),
                'colsample_bylevel': np.random.uniform(0.0, 1.0),
            }
        self.hyperparams = params
        return params

    def __repr__(self):
        return f'XGBPredictor(ss_type={self.ss_type}, encoding_type={self.encoding_type}, hyperparams={self.hyperparams})'

    def __str__(self):
        model_status = 'fitted' if self.model else 'not fitted'
        mean_str = f'{self.mean:.4f}' if self.mean is not None else 'None'
        std_str = f'{self.std:.4f}' if self.std is not None else 'None'
        hyperparams_str = ', '.join([
            f'{k}={v}' for k, v in self.hyperparams.items()
        ]) if self.hyperparams else 'default'
        str_repr = (f'XGBPredictor:\n'
                    f'  Model Status: {model_status}\n'
                    f'  Encoding Type: {self.encoding_type}\n'
                    f'  Search Space Type: {self.ss_type}\n'
                    f'  Model Architecture: {self.model}\n'
                    f'  Hyperparameters: {hyperparams_str}\n'
                    f'  Mean (train): {mean_str}\n'
                    f'  Std (train): {std_str}\n')
        return str_repr

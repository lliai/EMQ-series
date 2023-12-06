# File: RFPredictor.py
# Description:
#   Definition of the RFPredictor class used for predicting performance in NAS problems using a Random Forest model.
#   RFPredictor (inheriting from the Predictor class in predictor.py) provides functionality for
#   saving/loading model states, fitting the model, making predictions, and setting random hyperparameters.

# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0

import os
import pickle

import numpy as np
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestRegressor

from .predictor import Predictor


class RFPredictor(Predictor):

    def __init__(self,
                 ss_type='nasbench101',
                 encoding_type='adj_onehot',
                 random_state=None):
        super().__init__(ss_type=ss_type, encoding_type=encoding_type)
        self.std = None
        self.mean = None
        self.model = None
        self.hyperparams = None
        self.random_state = random_state
        self.default_hyperparams = {
            'n_estimators': 116,
            'max_features': 0.17055852159745608,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'bootstrap': False,
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
        """Fit the model to the given training data"""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        if self.model is None:
            self.model = RandomForestRegressor(
                random_state=self.random_state, **self.hyperparams)

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
                'n_estimators': int(loguniform(16, 128).rvs()),
                'max_features': loguniform(0.1, 0.9).rvs(),
                'min_samples_leaf': int(np.random.choice(19) + 1),
                'min_samples_split': int(np.random.choice(18) + 2),
                'bootstrap': False,
            }
        self.hyperparams = params
        return params

    def __repr__(self):
        return f'RFPredictor(ss_type={self.ss_type}, encoding_type={self.encoding_type}, random_state={self.random_state}), hyperparams={self.hyperparams})'

    def __str__(self):
        model_status = 'fitted' if self.model else 'not fitted'
        mean_str = f'{self.mean:.4f}' if self.mean is not None else 'None'
        std_str = f'{self.std:.4f}' if self.std is not None else 'None'
        hyperparams_str = ', '.join([
            f'{k}={v}' for k, v in self.hyperparams.items()
        ]) if self.hyperparams else 'default'
        str_repr = (f'RFPredictor:\n'
                    f'  Model Status: {model_status}\n'
                    f'  Encoding Type: {self.encoding_type}\n'
                    f'  Search Space Type: {self.ss_type}\n'
                    f'  Model Architecture: {self.model}\n'
                    f'  Hyperparameters: {hyperparams_str}\n'
                    f'  Mean (train): {mean_str}\n'
                    f'  Std (train): {std_str}\n')
        return str_repr

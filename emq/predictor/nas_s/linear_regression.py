# File: LRPredictor.py
# Description:
#     Definition of the LRPredictor class used for predicting accuracies in NAS problems.
#     LRPredictor (inherits from the Predictor class in predictor.py) provides functionality for saving/loading model states,
#     fitting the model, and making predictions using linear regression.

import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression

from .predictor import Predictor


class LRPredictor(Predictor):

    def __init__(self, ss_type='nasbench101', encoding_type='adj_onehot'):
        super().__init__(ss_type=ss_type, encoding_type=encoding_type)
        self.std = None
        self.mean = None
        self.model = None

    def save(self, file_path):
        """Save the predictor to a file"""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump((self.model, self.mean, self.std), f)

    def load(self, file_path):
        """Load the predictor from a file"""
        with open(file_path, 'rb') as f:
            self.model, self.mean, self.std = pickle.load(f)

    def fit(self, xtrain, ytrain):
        """Fit predictor to training data"""
        if self.model is None:
            self.model = LinearRegression()

        # Normalize the training data
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain = (ytrain - self.mean) / self.std

        self.model.fit(xtrain, ytrain)

        train_pred = self.model.predict(xtrain)
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error  # MAE

    def refit(self, new_archs, new_accs):
        """Refit the predictor using new architectures and accuracies"""
        self.fit(new_archs, new_accs)

    def predict(self, xtest):
        """Predict the accuracy of a list of architectures"""
        if self.model is None:
            raise RuntimeError('The model has not been fitted yet.')
        predictions = self.model.predict(xtest)
        # Denormalize the predictions
        return predictions * self.std + self.mean

    def __repr__(self):
        return f'LRPredictor(ss_type={self.ss_type}, encoding_type={self.encoding_type})'

    def __str__(self):
        model_status = 'fitted' if self.model else 'not fitted'
        mean_str = f'{self.mean:.4f}' if self.mean is not None else 'None'
        std_str = f'{self.std:.4f}' if self.std is not None else 'None'
        str_repr = (f'LinearPredictor:\n'
                    f'  Model Status: {model_status}\n'
                    f'  Encoding Type: {self.encoding_type}\n'
                    f'  Search Space Type: {self.ss_type}\n'
                    f'  Model Architecture: {self.model}\n'
                    f'  Mean (train): {mean_str}\n'
                    f'  Std (train): {std_str}\n')
        return str_repr

# File: predictor.py
# Description:
#   Definition of the generic Predictor class for use in NAS frameworks.
#   Provides the structure for creating, saving, loading, fitting, and evaluating predictors.
#   The class also supports setting and optimizing hyperparameters. It acts as a base class for
#   specific predictor implementations, allowing for flexibility and extensibility in NAS frameworks.
#   The predictor class is also capable of performing hyperparameter optimization using cross-validation and early stopping strategies.

# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0

import copy
import math
import time

import numpy as np
import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import cross_validation


class Predictor:

    def __init__(self, ss_type=None, encoding_type=None):
        """Predictor base class

        Keyword Arguments:
            ss_type -- search space type (default: {None})
            encoding_type -- encoding type (default: {None})
        """
        self.ss_type = ss_type
        self.encoding_type = encoding_type

    def set_ss_type(self, ss_type):
        """Set search space type

        Arguments:
            ss_type -- search space type
        """
        self.ss_type = ss_type

    def save(self, file_path):
        """Save predictor to file"""
        raise NotImplementedError(
            'Save method not implemented for this predictor')

    def load(self, file_path):
        """Load predictor from file"""
        raise NotImplementedError(
            'Load method not implemented for this predictor')

    def fit(self, xtrain, ytrain):
        """Fit predictor to training data
        This can be called any number of times during the NAS algorithm.

        input: list of architectures, list of architecture accuracies
        output: none

        Arguments:
            xtrain -- training data (architectures)
            ytrain -- training labels (accuracies)
        """
        pass

    def refit(self, xtrain_new, ytrain_new):
        """Refit the predictor with new training data.

        This can be called any number of times during the NAS algorithm.

        input: list of new architectures, list of new architecture accuracies
        output: none

        Arguments:
            xtrain_new -- new training data (architectures)
            ytrain_new -- new training labels (accuracies)
        """
        self.fit(xtrain_new, ytrain_new)

    def predict(self, xtest):
        """Query predictor for test data
        This can be called any number of times during the NAS algorithm.

        inputs: list of architectures,
        output: predictions for the architectures

        Arguments:
            xtest -- test data (architectures)

        Returns:
            predictions -- predictions for the architectures
        """
        pass

    def set_hyperparams(self, hyperparams):
        """Set hyperparameters

        Arguments:
            hyperparams -- hyperparameters
        """
        self.hyperparams = hyperparams

    def set_random_hyperparams(self):
        """Set random hyperparameters"""
        raise NotImplementedError(
            'Random hyperparameters not implemented for this predictor')

    def get_hyperparams(self):
        """Get hyperparameters

        Returns:
            hyperparams -- hyperparameters
        """
        if hasattr(self, 'hyperparams'):
            return self.hyperparams
        else:
            print('no hyperparams set')
            return None

    def reset_hyperparams(self):
        """Reset hyperparameters"""
        self.hyperparams = None

    def evaluate(self, X_test, y_test):
        """Evaluate predictor on test data

        Arguments:
            xtest -- test data (architectures)
            ytest -- test labels (accuracies)

        Returns:
            metrics -- dictionary of metrics
        """
        y_pred = self.predict(X_test)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'pearson': scipy.stats.pearsonr(y_test, y_pred)[0],
            'spearman': scipy.stats.spearmanr(y_test, y_pred)[0],
            'kendalltau': scipy.stats.kendalltau(y_test, y_pred)[0],
            'r2_score': r2_score(y_test, y_pred),
        }
        return metrics

    def run_hpo(self,
                xtrain,
                ytrain,
                start_time,
                metric='kendalltau',
                max_iters=5000,
                max_hpo_time=60,
                patience=50,
                k=3,
                verbose=True):
        """
        Run hyperparameter optimization.

        Arguments:
            xtrain -- training data (architectures)
            ytrain -- training labels (accuracies)
            start_time -- start time

        Keyword Arguments:
            metric -- metric used for optimization (default: {"kendalltau"})
            max_iters -- maximum number of iterations (default: {5000})
            max_hpo_time -- maximum time for hyperparameter optimization in seconds (default: {60})
            patience -- patience of early stopping (default: {50})
            k -- number of folds in k-fold cross validation (default: {3})
            verbose -- display progress and information (default: {False})

        Returns:
            best_hyperparams -- best hyperparameters
            best_score -- best score
        """

        if verbose:
            print('Starting cross validation')

        # Initialization
        best_score = -1e6
        best_hyperparams = None
        no_improvement_count = 0

        for iter_num in range(max_iters):
            # Set new random hyperparameters
            hyperparams = self.set_random_hyperparams()

            # Evaluate hyperparameters
            cv_score = cross_validation(xtrain, ytrain, self, k, metric)

            if verbose:
                print(
                    f'[{iter_num}/{max_iters}] cv_score={cv_score:.4f}, params={hyperparams}'
                )

            if np.isnan(cv_score) or cv_score < 0:
                cv_score = 0

            # Check if score is better than previous best
            if cv_score > best_score or iter_num == 0:
                best_hyperparams = hyperparams
                best_score = cv_score
                no_improvement_count = 0
                if verbose:
                    print(
                        f'--> new best score = {cv_score}, hparams = {hyperparams}'
                    )
            else:
                no_improvement_count += 1

            if verbose:
                print(f'patience = {no_improvement_count}/{patience}')

            # Calculate elapsed time and time limit
            elapsed_time = time.time() - start_time
            # Max HPO time is scaled by the number of architectures, with a minimum of 20 seconds
            time_limit = max_hpo_time * (len(xtrain) / 1000) + 20

            # Check if early stopping conditions are met
            if elapsed_time > time_limit or no_improvement_count > patience:
                if verbose:
                    print(f'Stopping HPO prematurely')
                    print(
                        f'Time = {elapsed_time} > {time_limit} or {no_improvement_count} > {patience}'
                    )
                break

        if math.isnan(best_score):
            best_hyperparams = self.default_hyperparams

        if verbose:
            print(f'Finished {iter_num+1} rounds')
            print(
                f'Best hyperparams = {best_hyperparams} Score = {best_score}')

        self.hyperparams = best_hyperparams

        return best_hyperparams.copy(), best_score

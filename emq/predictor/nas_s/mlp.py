# File: MLPPredictor.py
# Description:
#   Definition of the MLPPredictor and FeedforwardNet classes used for predicting performance in NAS problems.
#   MLPPredictor  (inherits from the Predictor class in predictor.py), handles tasks like
#   saving/loading model states, fitting the model, and making predictions.
#   FeedforwardNet represents a fully connected neural network with customizable layers and widths.
#   A utility function, accuracy_mse, is also provided for calculating the mean squared error.

# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import AverageMeterGroup, get_logger

from .predictor import Predictor

# NOTE: faster on CPU
device = torch.device('cpu')


def accuracy_mse(prediction, target, scale=100.0):
    """Computes the MSE between the prediction and the target"""
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class FeedforwardNet(nn.Module):

    def __init__(
        self,
        input_dims,
        num_layers=3,
        layer_width=[10, 10, 10],
        output_dims=1,
        activation='relu',
    ):
        super(FeedforwardNet, self).__init__()
        assert len(
            layer_width
        ) == num_layers, 'number of widths should be equal to the number of layers'

        self.activation = eval('F.' + activation)

        all_units = [input_dims] + layer_width

        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(all_units[i], all_units[i + 1])
            for i in range(num_layers)
        ])
        self.out = nn.Linear(all_units[-1], output_dims)

        # Initialize weights and biases (Xavier initialization)
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """Forward pass"""
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.out(x)


class MLPPredictor(Predictor):

    def __init__(self,
                 ss_type='nasbench101',
                 encoding_type='adj_onehot',
                 random_state=None):
        super().__init__(ss_type=ss_type, encoding_type=encoding_type)
        self.std = None
        self.mean = None
        self.model = None
        self.hyperparams = None
        self.default_hyperparams = {
            'num_layers': 20,
            'layer_width': [20] * 20,
            'batch_size': 32,
            'lr': 0.001,
            'regularization': 0.2,
            'epochs': 500,
        }

    def save(self, file_path):
        """Save the model and its parameters to the given file path."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'hyperparams': self.hyperparams,
        }
        torch.save(checkpoint, file_path)

    def load(self, path):
        """Load the model and parameters from a file."""
        checkpoint = torch.load(path)

        # Load hyperparams
        self.hyperparams = checkpoint['hyperparams']
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']

        # Load model
        num_layers = checkpoint['hyperparams']['num_layers']
        layer_width = checkpoint['hyperparams']['layer_width']
        input_dims = checkpoint['model_state_dict']['layers.0.weight'].shape[1]

        self.model = self.get_model(
            input_dims=input_dims,
            num_layers=num_layers,
            layer_width=layer_width,
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)

    def get_model(self, **kwargs):
        """Get model"""
        return FeedforwardNet(**kwargs)

    def fit(self, xtrain, ytrain):
        """Fit the model"""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams

        # Set hyperparams
        num_layers = self.hyperparams['num_layers']
        layer_width = self.hyperparams['layer_width']
        batch_size = self.hyperparams['batch_size']
        lr = self.hyperparams['lr']
        regularization = self.hyperparams['regularization']
        epochs = self.hyperparams['epochs']

        _xtrain = xtrain

        # Normalize train targets
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain = (ytrain - self.mean) / self.std

        _ytrain = np.array(ytrain)

        # Create data loader
        train_data = TensorDataset(
            torch.FloatTensor(_xtrain).to(device),
            torch.FloatTensor(_ytrain).to(device),
        )
        data_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)

        # Create model
        self.model = self.get_model(
            input_dims=_xtrain.shape[1],
            num_layers=num_layers,
            layer_width=layer_width,
        ).to(device)

        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        # Create loss function
        criterion = nn.MSELoss().to(device)

        # Logging setup
        logger = get_logger()

        # Set model to train mode
        self.model.train()

        # Start training
        for epoch in range(epochs):
            meters = AverageMeterGroup()
            for i, batch in enumerate(data_loader):
                optimizer.zero_grad()
                _input = batch[0].to(device)
                _target = batch[1].to(device)
                _prediction = self.model(_input).view(-1)

                loss = criterion(_prediction, _target)

                # Add L1 regularization
                params = torch.cat([
                    x[1].view(-1) for x in self.model.named_parameters()
                    if x[0] == 'out.weight'
                ])

                loss += regularization * torch.norm(params, 1)
                loss.backward()

                optimizer.step()

                _prediction_denorm = _prediction * self.std + self.mean
                _target_denorm = _target * self.std + self.mean
                mse = accuracy_mse(_prediction_denorm, _target_denorm)

                meters.update({
                    'loss': loss.item(),
                    'mse': mse.item()
                },
                              n=_target.size(0))

            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {meters['loss'].avg:.4f}")

        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(np.abs(train_pred - ytrain))

        return train_error

    def predict(self, xtest):
        """Predict the target values for the given test data."""
        test_data = TensorDataset(
            torch.FloatTensor(xtest).to(device),
            torch.zeros(len(xtest)).to(device))
        eval_batch_size = len(xtest)

        test_data_loader = DataLoader(
            test_data, batch_size=eval_batch_size, shuffle=False)

        self.model.eval()

        predictions = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                _prediction = self.model(batch[0].to(device)).view(-1)
                predictions.append(_prediction.cpu().numpy())

        predictions = np.concatenate(predictions)
        return predictions * self.std + self.mean  # Denormalize predictions

    def set_random_hyperparams(self):
        """Set random hyperparameters"""
        num_layers = np.random.randint(5, 25)
        params = {
            'num_layers': num_layers,
            'layer_width': [np.random.randint(5, 25)] * num_layers,
            'batch_size': 32,
            'lr': 10**np.random.uniform(-4, -1),
            'regularization': 0.2,
            'epochs': 500,
        }
        self.hyperparams = params
        return params

    def __repr__(self):
        repr_str = f'MLPPredictor(encoding_type={self.encoding_type}, ss_type={self.ss_type}, model={self.model}, hyperparams={self.hyperparams})'
        return repr_str

    def __str__(self):
        model_status = 'fitted' if self.model else 'not fitted'
        mean_str = f'{self.mean:.4f}' if self.mean is not None else 'None'
        std_str = f'{self.std:.4f}' if self.std is not None else 'None'
        hyperparams_str = ', '.join([
            f'{k}={v}' for k, v in self.hyperparams.items()
        ]) if self.hyperparams else 'default'
        str_repr = (f'MLPPredictor:\n'
                    f'  Model Status: {model_status}\n'
                    f'  Encoding Type: {self.encoding_type}\n'
                    f'  Search Space Type: {self.ss_type}\n'
                    f'  Model Architecture: {self.model}\n'
                    f'  Hyperparameters: {hyperparams_str}\n'
                    f'  Mean (train): {mean_str}\n'
                    f'  Std (train): {std_str}\n')
        return str_repr

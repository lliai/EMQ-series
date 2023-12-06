# File: gcn_predictor.py
# Description:
#   This file implements the GCNPredictor class, which is used for predicting
#   the performance of architectures in NAS problems using Graph Convolutional Networks (GCNs).
#   The GCNPredictor (inherits from the Predictor class defined in predictor.py) provides functionality
#   for saving/loading model states, fitting the model, and making predictions.
#
#   The GCNPredictor uses the NASBench101Dataset class from the dataset module (dataset.py) and
#   the AverageMeterGroup class from the utils module (utils.py).

# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0
# This is an implementation of gcn predictor for NAS from the paper:
# Wen et al., 2019. Neural Predictor for Neural Architecture Search

import itertools
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import NASBench101Dataset
from scipy.stats import loguniform
from torch.utils.data import DataLoader

from emq.utils.misc import AverageMeterGroup
from .predictor import Predictor

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    """Pool the node features of a graph by averaging over all nodes."""
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


def accuracy_mse(prediction, target, scale=100.0):
    """Computes the MSE between the prediction and the target"""
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


# GCN implementation from: https://github.com/ultmaster/neuralpredictor.pytorch
# License: MIT
class DirectedGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        """Forward pass of the layer"""
        norm_adj = normalize_adj(adj)
        # Convolution
        output1 = F.relu(
            torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        # Convolution
        output2 = F.relu(
            torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return (self.__class__.__name__ + ' (' + str(self.in_features) +
                ' -> ' + str(self.out_features) + ')')


# GCN implementation from: https://github.com/ultmaster/neuralpredictor.pytorch
# License: MIT
class NeuralPredictorModel(nn.Module):

    def __init__(self,
                 initial_hidden=5,
                 gcn_hidden=144,
                 gcn_layers=3,
                 linear_hidden=128):
        super().__init__()
        self.gcn = [
            DirectedGraphConvolution(initial_hidden if i == 0 else gcn_hidden,
                                     gcn_hidden) for i in range(gcn_layers)
        ]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(gcn_hidden, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

    def forward(self, inputs):
        """Forward pass"""
        numv, adj, out = (
            inputs['num_vertices'],
            inputs['adjacency'][0],
            inputs['operations'],
        )

        adj = adj.to(device)
        numv = numv.to(device)
        out = out.to(device)

        gs = adj.size(1)  # graph node number

        adj_with_diag = normalize_adj(
            adj +
            torch.eye(gs, device=adj.device))  # assuming diagonal is not 1

        # Neighboring aggregation
        for layer in self.gcn:
            out = layer(out, adj_with_diag)

        # Graph pooling
        out = graph_pooling(out, numv)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out


class GCNPredictor(Predictor):

    def __init__(self,
                 ss_type='nasbench101',
                 encoding_type='gcn_graph',
                 random_state=None):
        super().__init__(ss_type=ss_type, encoding_type=encoding_type)
        self.std = None
        self.mean = None
        self.model = None
        self.hyperparams = None
        self.default_hyperparams = {
            'gcn_hidden': 144,
            'batch_size': 10,
            'lr': 1e-4,
            'wd': 1e-3,
            'epochs': 300,
            'eval_batch_size': 1000
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
        # Load the model parameters from checkpoint
        self.hyperparams = checkpoint['hyperparams']
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        # Load the model
        gcn_hidden = checkpoint['hyperparams'][
            'gcn_hidden']  # Get gcn_hidden from the checkpoint
        self.model = NeuralPredictorModel(gcn_hidden=gcn_hidden)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)

    def fit(self, xtrain, ytrain):
        """Train the model on the given data"""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        # Set hyperparameters
        gcn_hidden = self.hyperparams['gcn_hidden']
        batch_size = self.hyperparams['batch_size']
        lr = self.hyperparams['lr']
        wd = self.hyperparams['wd']
        epochs = self.hyperparams['epochs']

        # Normalize the target values
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean) / self.std

        train_data = []
        for i, arch in enumerate(xtrain):
            arch['val_acc'] = float(ytrain_normed[i])
            train_data.append(arch)
        train_data = np.array(train_data)

        if self.model is None:
            self.model = NeuralPredictorModel(gcn_hidden=gcn_hidden)
        self.model.to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Set up logging
        logger = get_logger()

        # xtrain (train_data) is a list of dicts
        data_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        # Initialize AverageMeterGroup
        meters = AverageMeterGroup()

        # Set model to training mode
        self.model.train()

        # Start training
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']
            for _, batch in enumerate(data_loader):
                target = batch['val_acc'].to(device)
                prediction = self.model(batch)
                optimizer.zero_grad()
                loss = criterion(prediction.float(), target.float())
                loss.backward()
                optimizer.step()

                prediction_denorm = prediction * self.std + self.mean
                target_denorm = target * self.std + self.mean
                mse = accuracy_mse(prediction_denorm, target_denorm)
                # Update the meters object
                meters.update({
                    'loss': loss.item(),
                    'mse': mse.item()
                },
                              n=target.size(0))

            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {meters['loss'].avg:.4f}, MSE: {meters['mse'].avg:.4f}"
            )

            lr_scheduler.step()

        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error  # mean absolute error

    def refit(self, new_archs, new_accs):
        """Refit the model with new data"""
        self.fit(new_archs, new_accs)

    def predict(self, xtest):
        """Predict the accuracy of architectures"""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        eval_batch_size = self.hyperparams['eval_batch_size']
        test_data_loader = DataLoader(xtest, batch_size=eval_batch_size)

        # Set model to eval mode
        self.model.eval()
        meters = AverageMeterGroup()
        criterion = nn.MSELoss().to(device)
        # Set up logging
        logger = get_logger()

        predict_, target_ = [], []

        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                #batch = batch.to(device)
                target = batch['val_acc'].to(device)
                # Forward pass
                prediction = self.model(batch)
                predict_.append(prediction.cpu().numpy())
                target_.append(target.cpu().numpy())

                # Update the meters object
                loss = criterion(prediction.float(), target.float())
                _prediction_denormalized = prediction * self.std + self.mean
                _target_denormalized = target * self.std + self.mean
                mse = accuracy_mse(_prediction_denormalized,
                                   _target_denormalized)
                meters.update({
                    'loss': loss.item(),
                    'mse': mse.item()
                },
                              n=target.size(0))

                #logger.info(f"Batch {i + 1}, Loss: {meters['loss'].avg:.4f}, MSE: {meters['mse'].avg:.4f}")

        predict_ = np.concatenate(predict_)
        target_ = np.concatenate(target_)

        # NOTE: Model predicts the performance metrics in the normalized form
        # since it was trained on normalized target values
        return predict_ * self.std + self.mean  # predictions are denormalized

    def set_random_hyperparams(self):
        """Set random hyperparameters for the model"""
        params = {
            'gcn_hidden': int(loguniform(64, 200).rvs()),
            'batch_size': int(loguniform(5, 32).rvs()),
            'lr': loguniform(0.00001, 0.1).rvs(),
            'wd': loguniform(0.00001, 0.1).rvs(),
            'epochs': 200,
            'eval_batch_size': 1000
        }

        self.hyperparams = params
        return params

    def __str__(self):
        model_status = 'fitted' if self.model else 'not fitted'
        mean_str = f'{self.mean:.4f}' if self.mean is not None else 'None'
        std_str = f'{self.std:.4f}' if self.std is not None else 'None'
        hyperparams_str = ', '.join([
            f'{k}={v}' for k, v in self.hyperparams.items()
        ]) if self.hyperparams else 'default'
        str_repr = (f'GCNPredictor:\n'
                    f'  Model Status: {model_status}\n'
                    f'  Encoding Type: {self.encoding_type}\n'
                    f'  Search Space Type: {self.ss_type}\n'
                    f'  Model Architecture: {self.model}\n'
                    f'  Hyperparameters: {hyperparams_str}\n'
                    f'  Mean (train): {mean_str}\n'
                    f'  Std (train): {std_str}\n')
        return str_repr

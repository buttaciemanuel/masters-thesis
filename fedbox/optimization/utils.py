from ..datasets import utils

from abc import ABC, abstractmethod
import hashlib
import math
import numpy
import json
import os
import torch
import time
from typing import Callable


class WeightingScheme(ABC):
    '''
    This abstract class defines a weighting scheme to aggregate updates from agents.
    '''
    
    def __init__(self, datasets: dict[str, list[utils.FederatedSubset]]):
        '''
        Constructs an aggregation scheme using the list of clients' `datasets`.

        Parameters
        ----------
        datasets: dict[str, list[utils.FederatedSubset]]
            List of local subsets where each one corresponds to a client
        '''
        
        self.datasets_ = datasets
        self.weights_ = None

    @abstractmethod
    def weights(self) -> dict[str, list[float]]:
        '''
        Computes aggregation weights.

        Returns
        -------
        dict[str, list[float]]
            Aggregation weights
        '''
        
        pass


class UniformWeightingScheme(WeightingScheme):
    '''
    This class defines a uniform weighting scheme to aggregate updates from agents.
    '''
    
    def __init__(self, datasets: dict[str, list[utils.FederatedSubset]]):
        '''
        Constructs an aggregation scheme using the list of clients' `datasets`.

        Parameters
        ----------
        datasets: dict[str, list[utils.FederatedSubset]]
            List of local subsets where each one corresponds to a client
        '''

        super().__init__(datasets)

    def weights(self) -> dict[str, list[float]]:
        '''
        Computes uniform aggregation weights.

        Returns
        -------
        dict[str, list[float]]
            Uniform aggregation weights
        '''

        if self.weights_ is None:
            self.weights_ = {
                group: [ 1 / len(subsets) for _ in subsets ]
                for group, subsets in self.datasets_.items()
            }

        return self.weights_
    

class AdjancencyWeightingScheme(WeightingScheme):
    '''
    This class defines a custom weighting scheme to aggregate updates from agents based on
    mutual statistical similarities computed as the negative logarithm of client misalignments.

    Note
    ----
    Each weight corresponds to the degree of each agent as a node in the graph representation of
    the federated network.
    '''
    
    def __init__(self, datasets: dict[str, list[utils.FederatedSubset]]):
        '''
        Constructs an aggregation scheme using the list of clients' `datasets`.

        Parameters
        ----------
        datasets: dict[str, list[utils.FederatedSubset]]
            List of local subsets where each one corresponds to a client
        '''

        super().__init__(datasets)

        self.adjacency_matrices_ = None

    def weights(self) -> dict[str, list[float]]:
        '''
        Computes aggregation weights as clients' degrees in the graph representation of the network.

        Returns
        -------
        dict[str, list[float]]
            Aggregation weights as clients' degrees (adjacency matrix)
        '''

        if self.weights_ is None or self.adjacency_matrices_ is None:
            self.adjacency_matrices_ = {}
            self.weights_ = {} 
            
            for group, subsets in self.datasets_.items():
                self.adjacency_matrices_[group], self.weights_[group] = self.adjacency_matrix_(subsets)

        return self.weights_
    
    def adjacencies(self) -> dict[str, torch.Tensor]:
        '''
        Computes the adjacency matrices for the training network (training clients) and testing network (testing clients).

        Returns
        -------
        dict[str, torch.Tensor]
            Adjacency matrices for training clients ('training') and testing clients ('testing')
        '''

        if self.weights_ is None or self.adjacency_matrices_ is None:
            self.weights()

        return self.adjacency_matrices_

    def adjacency_matrix_(self, subsets: list[utils.FederatedSubset]) -> tuple[torch.Tensor, list[float]]:
        '''
        Computes the adjacency matrix and related aggregation weights on a network of clients where each client corresponds to one subset in `subsets`.

        Parameters
        ---------
        subsets: list[utils.FederatedSubset]
            List of clients' subsets

        Returns
        -------
        tuple[torch.Tensor, list[float]]
            Tuple containing the adjacency matrix built on statistical similarities (0) and clients' aggregation weights as degrees in the related graph (1)
        '''

        n_subsets = len(subsets)
        messages = []

        adjacency_matrix = torch.zeros(size = (n_subsets, n_subsets))

        for subset in subsets:
            messages.append(self.message_(subset))

        for i in range(n_subsets):
            for j in range(i + 1, n_subsets):
                # client misalignment for couple of clients (i, j)
                misalignment = (1 - torch.dot(messages[i], messages[j])) / 2
                # connection weight in the adjacency matrix as the negative logarithm of the misalignment
                adjacency_matrix[i][j] = adjacency_matrix[j][i] = -torch.log(misalignment)

        # aggregation weights are computed as node degrees from the adjacency matrix of the graph
        weights = torch.sum(adjacency_matrix, dim = 1).tolist()

        return adjacency_matrix, weights
    
    def message_(self, subset: utils.FederatedSubset) -> torch.Tensor:
        '''
        Computes the statistically-significant message on a client's local subset.

        Parameters
        ----------
        subset: utils.FederatedSubset
            Subset of samples of a specific client

        Returns
        -------
        torch.Tensor
            Unitary vector message, i.e. normalized first principal component

        Note
        ----
        The message is computed as the first principal component on the client's dataset.
        '''

        if isinstance(subset.data, torch.Tensor):
            data = subset.data.view(len(subset), -1)
        else:
            data = subset.data.reshape(len(subset), -1)
            data = torch.tensor(data)

        message = torch.zeros(size = (data.shape[1],))
        # the statistically-significant message is computed as the first
        # principal component of the local dataset using the unscaled and
        # uncentered principal component analysis
        x = data.reshape(len(subset.data), -1) / subset.normalization
        _, _, vh = torch.linalg.svd(x, full_matrices = False)
        component = vh[0]
        rank = len(component)
        # the message is normalized as a unitary vector
        message[:rank] = component
        message /= message.norm()

        return message
    
class Logger:
    '''
    This class allows to log simulation metrics on a JSON file.
    '''

    def __init__(self, directory: str, simulation: dict, convergence: Callable, enable: bool = True):
        '''
        Constructs a logger instance.

        Parameters
        ----------
        directory: str
            Directory where to save the produced JSON file
        simulation: dict
            Simulation metadata
        convergence: Callable
            Boolean predicate that is run at each `log(...)` call to detect if the model converged
        enable: bool
            Flag to enable logging, `True` by default
        '''
        
        self.directory: str = directory
        self.simulation: dict = simulation
        self.metrics: list[dict] = []
        self.convergence_time: int = numpy.inf
        self.execution_time = time.perf_counter()
        self.convergence = convergence
        self.enable = enable

    def log(self, values: dict):
        '''
        Adds new logged data to the buffer and checks whether convergence has been reached.

        Parameters
        ----------
        values: dict
            Dictionary of logged values
        '''
        
        self.metrics.append(values)

        if numpy.isfinite(self.convergence_time):
            return
        
        converged = self.convergence(values)

        if converged is not None:
            self.convergence_time = converged

    def flush(self):
        '''
        Flushes the buffer's content to the output JSON file.
        '''

        if not self.enable:
            return
        
        seconds = math.floor(time.perf_counter() -  self.execution_time)

        if seconds < 60:
            self.execution_time = '{:02}s'.format(seconds)
        elif seconds < 3600:
            self.execution_time = '{:02}m{:02}s'.format(seconds // 60, seconds % 60)
        elif seconds < 86400:
            self.execution_time = '{:02}h{:02}m'.format(seconds // 3600, (seconds % 3600) // 60)
        else:
            self.execution_time = '{}d{:02}h'.format(seconds // 86400, (seconds % 86400) // 3600)
        
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        data = { **self.simulation, 'execution_time': self.execution_time, 'convergence_time': self.convergence_time, 'metrics': self.metrics }

        with open(os.path.join(self.directory, '{}.json'.format(hashlib.sha1(str(self.simulation.items()).encode()).hexdigest())), 'w', encoding = 'utf8') as file:
            json.dump(data, file, indent = 4)

    @staticmethod
    def default():
        '''
        Constructs a default fake logger.

        Returns
        -------
        Logger
            Fake logger
        '''
        
        return Logger('', '', lambda _: (), False)
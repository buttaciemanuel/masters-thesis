from ..datasets import utils
from . import fedavg
from .utils import WeightingScheme, AdjancencyWeightingScheme, Logger

from collections import OrderedDict
from copy import deepcopy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm


class Agent(fedavg.Agent):
    '''
    An agent (client) uses our novel scheme to optimize a shared model on its local subset.
    '''

    def __init__(self, subset: utils.FederatedSubset):
        '''
        Initializes the agent with a local `subset` of data samples and labels.

        Parameters
        ----------
        subset: utils.FederatedSubset
            Subset of data samples and labels
        '''
        
        self.subset = subset

    def step(self, beta: float, u: torch.nn.Module, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer: optim.Optimizer, max_gradient_norm: float):
        '''
        Performs an optimization step on `model` using minibatch (`x`, `y`) by introducing our `beta`-specific perturbation.

        Parameters
        ----------
        beta: float
            Parameter of our algorithm that controls the perturbation (the lower beta, the higher the perturbation)
        u: torch.nn.Module
            Averaged model with the latest updates (previous round) weighted by mutual statistical similarities with other agents
        model: torch.nn.Module
            Model that is optimized locally
        x: torch.Tensor
            Data samples in the minibatch
        y: torch.Tensor
            Data labels in the minibatch
        optimizer: optim.Optimizer
            Gradient-based optimizer
        max_gradient_norm: float
            Value used to clip the norm of the stochastic gradient
        '''
        
        perturbed = self.perturb(beta, u, model)
        prediction = perturbed(x)
        
        loss = nn.functional.cross_entropy(prediction, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(perturbed.parameters(), max_gradient_norm, error_if_nonfinite = True)

        model.load_state_dict(OrderedDict([
            (name, model.get_parameter(name) - optimizer.param_groups[-1]['lr'] * perturbed.get_parameter(name).grad.detach())
            for name in model.state_dict().keys()
        ]))

    def optimize(self, beta: float, u: torch.nn.Module, model: torch.nn.Module, n_steps: int, step_size: float, l2_penalty: float, max_gradient_norm: float, device: torch.device):
        '''
        Runs `n_steps` stochastic gradient descent steps while injecting `beta`-specific perturbation on the local dataset (one step for each minibatch).

        Parameters
        ----------
        beta: float
            Parameter of our algorithm that controls the perturbation (the lower beta, the higher the perturbation)
        u: torch.nn.Module
            Averaged model with the latest updates (previous round) weighted by mutual statistical similarities with other agents
        model: torch.nn.Module
            Model that is locally optimized
        n_steps: int
            Number of local SGD steps, i.e. number of minibatches
        step_size: float
            Step size or learning rate
        l2_penalty: float
            Weight of L2 (Tikhonov) regularization term
        max_gradient_norm: float
            Value used to clip the norm of the stochastic gradient
        device: torch.device
            Accelerator to run the code
        '''

        loader = data.DataLoader(self.subset, batch_size = len(self.subset) // n_steps)
        optimizer = optim.SGD(model.parameters(), lr = step_size, weight_decay = l2_penalty)

        model.train()

        for x, y in loader:
            self.step(beta, u, model, x.to(device), y.to(device), optimizer = optimizer, max_gradient_norm = max_gradient_norm)

        return model

    def multioptimize(self, beta: float, u: torch.nn.Module, model: torch.nn.Module, n_epochs: int, batch_size: int, step_size: float, l2_penalty: float, max_gradient_norm: float, device: torch.device):
        '''
        Runs `n_epochs` stochastic gradient descent epochs while injecting `beta`-specific perturbation on the local dataset.

        Parameters
        ----------
        beta: float
            Parameter of our algorithm that controls the perturbation (the lower beta, the higher the perturbation)
        u: torch.nn.Module
            Averaged model with the latest updates (previous round) weighted by mutual statistical similarities with other agents
        model: torch.nn.Module
            Model that is locally optimized
        n_epochs: int
            Number of local epochs to pass over the entire local dataset
        step_size: float
            Step size or learning rate
        l2_penalty: float
            Weight of L2 (Tikhonov) regularization term
        max_gradient_norm: float
            Value used to clip the norm of the stochastic gradient
        device: torch.device
            Accelerator to run the code

        Note
        ----
        Differently from `optimize(...)`, each epoch corresponds to passing over the entire dataset using SGD.
        '''

        loader = data.DataLoader(self.subset, batch_size = batch_size)
        optimizer = optim.SGD(model.parameters(), lr = step_size, weight_decay = l2_penalty)

        model.train()

        for _ in range(n_epochs):
            for x, y in loader:
                self.step(beta, u, model, x.to(device), y.to(device), optimizer = optimizer, max_gradient_norm = max_gradient_norm)

        return model

    def evaluate(self, model: torch.nn.Module, device: torch.device) -> tuple[float, float]:
        '''
        Evaluate the `model` by computing the average sample loss and accuracy.

        Parameters
        ----------
        model: torch.nn.Module
            Model that is locally optimized
        device: torch.device
            Accelerator to run the code

        Returns
        -------
        tuple[float, float]
            Tuple of average sample loss and accuracy on the local dataset
        '''

        loader = data.DataLoader(self.subset, batch_size = len(self.subset))
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device)

        model.eval()

        with torch.no_grad():
            prediction = model(x)
            loss = nn.functional.cross_entropy(prediction, y)
            accuracy = torch.sum(torch.argmax(prediction, dim = 1) == y)
            return loss.item(), accuracy.item()
        
    def perturb(self, beta: float, u: torch.nn.Module, model: torch.nn.Module) -> torch.nn.Module:
        '''
        Computes the perturbed model as `beta`-weighted average between `model` and `u`.

        Parameters
        ----------
        beta: float
            Parameter of our algorithm that controls the perturbation (the lower beta, the higher the perturbation)
        u: torch.nn.Module
            Averaged model with the latest updates (previous round) weighted by mutual statistical similarities with other agents
        model: torch.nn.Module
            Model that is locally optimized

        Returns
        -------
        torch.nn.Module
            Perturbed model as `beta`-weighted average between `model` and `u`
        '''
        
        perturbed_model = deepcopy(model)
        perturbed_model.load_state_dict(OrderedDict([
            (name, beta * model.state_dict()[name].detach() + (1 - beta) * u.state_dict()[name].detach())
            for name in model.state_dict().keys()
        ]))
        return perturbed_model


class Coordinator(fedavg.Coordinator):
    '''
    This class represents a centralized server coordinating the training of a shared model across multiple agents (i.e. clients).
    
    Note
    ----
    The agents locally update their models using our novel algorithmic framework.
    '''

    def __init__(
        self,
        beta: float,
        model: torch.nn.Module,
        datasets: dict[str, list[utils.FederatedSubset]],
        scheme: WeightingScheme,
        logger: Logger = Logger.default()
    ):
        '''
        Constructs the centralized coordinator, i.e. server, in the federated learning simulation.

        Parameters
        ----------
        beta: float
            Parameter controlling the perturbation (the lower beta, the higher the perturbation) while running our novel algorithm on clients
        model: torch.nn.Module
            Initial shared model
        datasets: dict[str, list[utils.FederatedSubset]]
            Training clients' subsets ('training') and testing clients' subsets ('testing')
        scheme: WeightingScheme
            Aggregation scheme to weight local updates from clients
        logger: Logger
            Logger instance to save progress during the simulation
        '''

        assert isinstance(scheme, AdjancencyWeightingScheme)

        assert beta > 0 and beta < 1

        self.beta = beta
        self.datasets = datasets
        self.model = model
        self.agents = {
            group: [ Agent(subset) for subset in dataset ] for group, dataset in datasets.items() 
        }
        self.weights = scheme.weights()
        self.adjacency_matrices = scheme.adjacencies()
        self.logger = logger

    def run(self, n_iterations: int, n_steps: int = None, n_epochs = None, batch_size: int = 32, step_size: float = 1e-3, step_size_diminishing: bool = False, l2_penalty: float = 1e-4, max_gradient_norm: float = 1.0, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''
        Runs `n_iterations` optimization (with our novel algorithm) and evaluation rounds on training clients.

        Parameters
        ----------
        n_iterations: int
            Number of global rounds
        n_steps: int
            Number of local SGD steps used for optimization on clients
        n_epochs: int
            Number of local epochs used for optimization on clients (mutually excludes n_steps)
        batch_size: int
            Number of samples in one SGD minibatch
        step_size: float
            Learning rate
        step_size_diminishing: bool
            This enables diminishing the step size linearly in time
        l2_penalty: float
            Weight of the L2 (Tikhonov) regularization used to penalize local models
        max_gradient_norm: float
            Value used to clip the norm of the stochastic gradient during local optimization
        device: torch.device
            Accelerator to run the code
        evaluate: bool
            Flag that enables evaluation of the update global model on training and testing clients

        Note
        ----
        Runs `n_iterations` times function `iterate(...)`.
        '''

        assert n_steps is not None or n_epochs is not None

        self.model = self.model.to(device)
        self.model.compile()
        self.u = [ deepcopy(self.model) for _ in self.agents['training']  ]

        for iteration in range(n_iterations):
            step_size_updated = step_size if not step_size_diminishing else step_size / (iteration + 1)
            metrics = self.iterate(iteration, n_steps, n_epochs, batch_size, step_size_updated, l2_penalty, max_gradient_norm, device, evaluate = True)
            
            self.logger.log({
                'step': iteration,
                'loss.training': metrics['training']['loss'],
                'loss.testing': metrics['testing']['loss'],
                'accuracy.training': metrics['training']['accuracy'],
                'accuracy.testing': metrics['testing']['accuracy'],
            })
            
            print(iteration, metrics)

    def iterate(self, iteration: int, n_steps: int, n_epochs: int, batch_size: int, step_size: float, l2_penalty: float, max_gradient_norm: float, device: torch.device, evaluate: bool = False) -> dict[str, float]:
        '''
        Runs a single optimization round with our novel algorithm on all training clients.

        Parameters
        ----------
        iteration: int
            Current global round
        n_steps: int
            Number of local SGD steps used for optimization on clients
        n_epochs: int
            Number of local epochs used for optimization on clients (mutually excludes n_steps)
        batch_size: int
            Number of samples in one SGD minibatch
        step_size: float
            Learning rate
        l2_penalty: float
            Weight of the L2 (Tikhonov) regularization used to penalize local models
        max_gradient_norm: float
            Value used to clip the norm of the stochastic gradient during local optimization
        device: torch.device
            Accelerator to run the code
        evaluate: bool
            Flag that enables evaluation of the update global model on training and testing clients

        Returns
        -------
        dict[str, float]
            Dictionary of current round's metrics
        '''

        indices = list(range(0, len(self.agents['training'])))
        k = len(self.agents['training'])
        
        random.shuffle(indices)
        
        indices = indices[:k]
        participants = [ self.agents['training'][i] for i in indices ]
        weights = [ self.weights['training'][i] for i in indices ]
        u = [ self.u[i] for i in indices ]

        initial_model = deepcopy(self.model)
        updates: list[nn.Module] = [ initial_model for _ in self.agents['training'] ]

        if n_steps is not None:
            for i, participant, u in tqdm(zip(indices, participants, self.u), total = len(participants), desc = 'Optimization on training agents (iteration {})'.format(iteration)):
                updates[i] = participant.optimize(self.beta, u, deepcopy(initial_model), n_steps = n_steps, step_size = step_size, l2_penalty = l2_penalty, max_gradient_norm = max_gradient_norm, device = device)
        else:
            for i, participant, u in tqdm(zip(indices, participants, self.u), total = len(participants), desc = 'Optimization on training agents (iteration {})'.format(iteration)):
                updates[i] = participant.multioptimize(self.beta, u, deepcopy(initial_model), n_epochs = n_epochs, batch_size = batch_size, step_size = step_size, l2_penalty = l2_penalty, max_gradient_norm = max_gradient_norm, device = device)

        self.average(updates, weights = weights)

        for i, degree in enumerate(self.weights['training']):
            # for each client, this computes the average of latest updates from other clients weighted accordingly to the similarity measure (normalized by client's degree)
            self.u[i].load_state_dict(OrderedDict([
                (
                    name, 
                    torch.stack([ 
                        similarity * update.state_dict()[name].detach()
                        for j, (update, similarity) in enumerate(zip(updates, self.adjacency_matrices['training'][i]))
                        if j != i
                    ]).sum(dim = 0) / degree
                )
                for name in self.model.state_dict().keys()
            ]))

        if not evaluate:
            return {}
        
        return self.evaluate(iteration, device)

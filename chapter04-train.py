'''
This file allows you to run the simulations for your federated learning experiments.

Examples
--------
The following command runs a simulation with `FedAvg` optimization algorithm on the `FEMNIST` dataset. 
Moreover, the partitioning among clients is IID, namely no heterogeneity, and the loss is strongly convex, 
that is a multinomial logistic regression. In addition, we set the number of rounds to 200, 
the local epochs to 1 and the aggregation scheme to `Adjacency`.
>>> python chapter04-train.py --dataset femnist \
>>>     --n_iterations 200 \
>>>     --n_epochs 1 \
>>>     --algorithm fedavg \
>>>     --weighting_scheme adjacency \
>>>     --output_directory simulations \
>>>     --log
'''

import argparse
import torch
import torchvision.transforms as transforms
import os

from fedbox.datasets import cifar10, cifar100, femnist
from fedbox.datasets.utils import set_seed
from fedbox.optimization.utils import UniformWeightingScheme, AdjancencyWeightingScheme
from fedbox.models.mlr import MultinomialLogisticRegression
from fedbox.models.nn import NeuralNetwork
from fedbox.optimization.utils import Logger
import fedbox.optimization.fedavg as fedavg
import fedbox.optimization.fedprox as fedprox
import fedbox.optimization.ours as ours

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
directory = './data'

parser = argparse.ArgumentParser('simulator', usage = 'Run simulations on federated datasets.')
parser.add_argument('-d', '--dataset', type = str, required = True, choices = [ 'cifar10', 'cifar100', 'femnist' ])
parser.add_argument('-t', '--n_iterations', type = int, default = 100)
parser.add_argument('-l', '--n_steps', type = int, default = None)
parser.add_argument('-e', '--n_epochs', type = int, default = 10)
parser.add_argument('-b', '--batch_size', type = int, default = 256)
parser.add_argument('-c', '--n_agents', type = int, default = 100)
parser.add_argument('-k', '--f_selected_agents', type = int, default = 1.0)
parser.add_argument('-s', '--step_size', type = float, default = 1e-3)
parser.add_argument('-sd', '--step_size_diminishing', action = 'store_true', default = False)
parser.add_argument('-g', '--max_gradient_norm', type = float, default = torch.inf)
parser.add_argument('-l2', '--l2_penalty', type = float, default = 1e-4)
parser.add_argument('-dv', '--device', type = str, choices = [ 'cpu', 'cuda' ], default = 'cpu')
parser.add_argument('-a', '--algorithm', required = True, type = str, choices = [ 'fedavg', 'fedprox', 'ours' ])
parser.add_argument('-r', '--random_seed', default = 0, type = int)
parser.add_argument('-w', '--weighting_scheme', required = True, choices = [ 'uniform', 'adjacency' ])
parser.add_argument('-pa', '--alpha', default = 0.1, type = float)
parser.add_argument('-pb', '--beta', default = 0.95, type = float)
parser.add_argument('-di', '--data_imbalance', default = 0.0, type = float)
parser.add_argument('-ci', '--class_imbalance', default = 0.0, type = float)
parser.add_argument('-nc', '--nonconvex', action = 'store_true', default = False)
parser.add_argument('-cr', '--convergence_rounds_threshold', type = float, default = 0.75)
parser.add_argument('-L', '--log', action = 'store_true', default = False)
parser.add_argument('-o', '--output_directory', type = str, default = './simulations')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.convergence_rounds_threshold < 0 or args.convergence_rounds_threshold > 1:
        print('error: argument \'convergence_rounds_threshold\' must be in range [0.0, 1.0]')
        exit(1)

    logger = Logger(
        directory = args.output_directory, 
        simulation = vars(args), 
        convergence = lambda values: values['step'] if values['accuracy.testing'] >= args.convergence_rounds_threshold else None, 
        enable = args.log
    )
    
    set_seed(args.random_seed)

    if args.dataset.upper() == 'CIFAR10':
        n_inputs = 3072
        n_classes = 10
        datasets = cifar10(
            directory, 
            n_subsets = args.n_agents,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            heterogeneity_degree = args.class_imbalance,
            samples_variance = args.data_imbalance,
            strict = False,
            seed = args.random_seed
        )
    elif args.dataset.upper() == 'CIFAR100':
        n_inputs = 3072
        n_classes = 100
        datasets = cifar100(
            directory, 
            n_subsets = args.n_agents,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            heterogeneity_degree = args.class_imbalance,
            samples_variance = args.data_imbalance,
            strict = False,
            seed = args.random_seed
        )
    elif args.dataset.upper() == 'FEMNIST':
        n_inputs = 784
        n_classes = 10
        datasets = femnist(
            directory, 
            n_subsets = args.n_agents,
            transform = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,))
            ]),
            heterogeneity_degree = args.class_imbalance,
            samples_variance = args.data_imbalance,
            strict = False,
            seed = args.random_seed
        )

    if args.nonconvex:
        model = NeuralNetwork(n_inputs = n_inputs, n_classes = n_classes)
    else:
        model = MultinomialLogisticRegression(n_inputs = n_inputs, n_classes = n_classes)

    scheme = None

    if args.weighting_scheme.lower() == 'uniform':
        scheme = UniformWeightingScheme(datasets)
    elif args.weighting_scheme.lower() == 'adjacency':
        scheme = AdjancencyWeightingScheme(datasets)

    if args.algorithm.lower() == 'fedavg':
        coordinator = fedavg.Coordinator(model, datasets, scheme, logger)
    elif args.algorithm.lower() == 'fedprox':
        coordinator = fedprox.Coordinator(args.alpha, model, datasets, scheme, logger)
    elif args.algorithm.lower() == 'ours':
        coordinator = ours.Coordinator(args.beta, model, datasets, scheme, logger)

    if args.n_steps is not None:
        coordinator.run(n_iterations = args.n_iterations, n_steps = args.n_steps, step_size = args.step_size, l2_penalty = args.l2_penalty, max_gradient_norm = args.max_gradient_norm, step_size_diminishing = args.step_size_diminishing)
    else:
        coordinator.run(n_iterations = args.n_iterations, n_epochs = args.n_epochs, batch_size = args.batch_size, step_size = args.step_size, l2_penalty = args.l2_penalty, max_gradient_norm = args.max_gradient_norm, step_size_diminishing = args.step_size_diminishing)

    logger.flush()
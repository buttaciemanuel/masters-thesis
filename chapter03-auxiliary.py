import argparse
import itertools
import matplotlib.pyplot as plt
import numpy
import os
import pandas as pd
import random
import seaborn
import torch
from typing import Tuple
import tqdm
import shutil

from fedbox.datasets import cifar10, cifar100, femnist, FederatedSubset

if shutil.which('latex'):
    params = {
        'ytick.color' : 'black',
        'xtick.color' : 'black',
        'text.usetex' : True,
        'font.family' : 'serif',
        'font.serif' : [ 'Times New Roman' ],
        'text.latex.preamble': '\\usepackage{times}',
        'legend.frameon': True,
    }
else:
    params = {
        'ytick.color' : 'black',
        'xtick.color' : 'black',
        'legend.frameon': True,
    }

palette = 'Set1'

plt.rcParams.update(params)

n_clients = 100

n_classes_of = {
    'cifar10': 10,
    'cifar100': 100,
    'femnist': 10
}

loaders = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'femnist': femnist
}

normalization = {
    'cifar10': 255,
    'cifar100': 255,
    'femnist': 1
}

features = {
    'cifar10': 3072,
    'cifar100': 3072,
    'femnist': 784
}

grid = {
    'dataset': [ 'cifar10', 'cifar100', 'femnist' ],
    'class_imbalance': [ None, 1, 10, 100 ],
    'data_imbalance': [ None, 1 ],
    'seed': [ 0, 1, 41 ]
}

flattened_grid = list(dict(zip(grid, x)) for x in itertools.product(*grid.values()))

def seed(s: int):
    random.seed(s)
    numpy.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)

def compute_class_distributions(partition: list[FederatedSubset], n_classes: int) -> numpy.ndarray:
    distributions = numpy.zeros((len(partition), n_classes))
    
    for i, subset in tqdm.tqdm(enumerate(partition)):
        labels, counts = numpy.unique(subset.targets, return_counts = True)
        distributions[i, labels.astype(int)] = counts

    return distributions

def compute_top_component(partition: list[FederatedSubset], normalization: float, n_features: int) -> numpy.ndarray:
    n_subsets = len(partition)
    components = torch.zeros((n_subsets, n_features))
    svd = torch.linalg.svd

    for i, subset in tqdm.tqdm(enumerate(partition)):
        x = torch.tensor(subset.data).reshape(len(subset.data), -1) / normalization
        u, s, vh = svd(x, full_matrices = False)
        component = vh[0]
        rank = len(component)
        components[i, :rank] = component

    return components.numpy()

def compute_misalignments(components: numpy.ndarray) -> list[Tuple[str, float]]:
    n_subsets = len(components)
    misalignments_list = []

    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            misalignment = (1 - numpy.dot(components[i], components[j])) / 2
            misalignments_list.append((misalignment, f'misalignment', i, j))

    return misalignments_list

parser = argparse.ArgumentParser('visualize', usage = 'Visualize auxiliary results.')
parser.add_argument('-o', '--output', type = str, default = './visualizations')

if __name__ == '__main__':
    args = parser.parse_args()
    misalignments = None
    # compute misalignments between each pair for each configuration
    for params in flattened_grid:
        datasets = loaders[params['dataset']]('./data', n_subsets = n_clients, heterogeneity_degree = params['class_imbalance'], samples_variance = params['data_imbalance'], seed = params['seed'], strict = False)
        subsets = datasets['training']
        components = compute_top_component(subsets, normalization = normalization[params['dataset']], n_features = features[params['dataset']])
        misalignments_list = compute_misalignments(components)

        if misalignments is None:
            misalignments = pd.DataFrame([ dict(**params, misalignment = x[0], metric = x[1], i = x[2], j = x[3]) for x in misalignments_list ])
        else:
            misalignments = pd.concat((
                misalignments,
                pd.DataFrame([ dict(**params, misalignment = x[0], metric = x[1], i = x[2], j = x[3]) for x in misalignments_list ])
            ), axis = 0)

    misalignments['misalignment'] = misalignments['misalignment'].astype(float)
    misalignments['data_imbalance'] = misalignments['data_imbalance'].astype(float)
    misalignments['class_imbalance'] = misalignments['class_imbalance'].astype(float)
    misalignments['dataset'] = misalignments['dataset'].astype('string')
    misalignments['metric'] = misalignments['metric'].astype('string')
    misalignments['data_imbalance'].fillna(0.0, inplace = True)
    misalignments['class_imbalance'].fillna(0.0, inplace = True)
    misalignments.sort_values([ 'dataset', 'class_imbalance', 'data_imbalance', 'metric' ], inplace = True)
    misalignments.reset_index(drop = True, inplace = True)
    misalignments['dataset'] = misalignments['dataset'].apply(lambda x: x.upper())
    
    # display distribution of misalignments using a violinplot
    fgrid = seaborn.FacetGrid(misalignments, col = 'dataset', margin_titles = True, sharey = False)
    fgrid.map_dataframe(seaborn.violinplot, x = 'class_imbalance', y = 'misalignment', hue = 'data_imbalance', split = True, cut = 0, linewidth = 1, inner = 'quart', gap = 0.1, log_scale = True, palette = palette)
    fgrid.add_legend()
    fgrid.legend.set_title('data imbalance')
    fgrid.set_ylabels('mis(i, j)')
    fgrid.set_xlabels('class imbalance')
    fgrid.set_titles(col_template = '{col_name}', fontweight = 'black')
    fgrid.tight_layout()

    fgrid.axes[0, 0].set_xlabel('')
    fgrid.axes[0, 2].set_xlabel('')

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    fgrid.savefig(os.path.join(args.output, 'client-misalignments.pdf'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    # computes the laplacian eigenvalues for each configuration of (dataset, class_imbalance, data_imbalance, seed)
    misalignments['weight'] = -numpy.log(misalignments['misalignment'])
    groups = misalignments.copy().groupby([ 'dataset', 'class_imbalance', 'data_imbalance', 'seed' ])
    eps = 1e-3
    eigenvalues = None

    for params, group in groups:
        nodes_count = 1 + group['i'].unique().shape[0]
        indices = group[[ 'i', 'j' ]].values
        values = group[[ 'weight' ]].values
        W = numpy.zeros((nodes_count, nodes_count))
        
        for (i, j), weight in zip(indices, values.ravel()):
            W[i, j] = weight
            W[j, i] = weight

        D = numpy.diag(W.sum(axis = 1))
        L = D - W

        eigenvalues_laplacian, eigenvectors_laplacian = numpy.linalg.eigh(L)
        eigenvalues_laplacian = numpy.where(eigenvalues_laplacian < eps, 0, eigenvalues_laplacian)
        
        frame = pd.DataFrame({ 
            'eigenvalue': eigenvalues_laplacian,
            'eigenvalue-index': numpy.arange(len(eigenvalues_laplacian))
        })
        frame['dataset'] = params[0]
        frame['class_imbalance'] = params[1]
        frame['data_imbalance'] = params[2]
        frame['seed'] = params[3]

        if eigenvalues is None:
            eigenvalues = frame
        else:
            eigenvalues = pd.concat((eigenvalues, frame), axis = 0)
        
    # display the laplacian spectrum for each configuration
    fgrid = seaborn.displot(eigenvalues, kind = 'kde', x = 'eigenvalue', hue = 'class_imbalance', log_scale = False, fill = True, row = 'data_imbalance', col = 'dataset', cut = 0, bw_adjust = 0.5, palette = palette, facet_kws = { 'margin_titles': True, 'sharex': True, 'sharey': True })

    fgrid.set_xlabels('laplacian spectrum')
    fgrid.set_titles(col_template = '{col_name}', row_template = 'data imbalance = ${row_name}$')

    fgrid.set_yticklabels(None)

    fgrid.axes[1, 0].set_xlabel('')
    fgrid.axes[1, 2].set_xlabel('')

    for ax in fgrid.axes_dict.values():
        ax.spines['left'].set_visible(False)
        ax.tick_params(left = False, labelbottom = True, bottom = True)

    fgrid.legend.set_title('class imbalance')

    fgrid.tight_layout()

    fgrid.savefig(os.path.join(args.output, 'laplacian-spectrum-distribution.pdf'), dpi = 300, bbox_inches = 'tight')

    plt.show()

    seaborn.set_palette(palette)

    # display class distribution for each client for each dataset in a balanced setting
    grid_for_distributions = {
        'data_imbalance': [ None ],
        'class_imbalance': [ None, 1, 10 ],
        'seed': [ 0 ]
    }

    flattened_grid_for_distributions = list(dict(zip(grid_for_distributions, x)) for x in itertools.product(*grid_for_distributions.values()))

    for dataset in [ 'cifar10', 'cifar100', 'femnist' ]:
        fig, axes = plt.subplots(1, 3)

        for params, ax in zip(flattened_grid_for_distributions, numpy.ravel(axes)):
            datasets = loaders[dataset]('./data', heterogeneity_degree = params['class_imbalance'], samples_variance = params['data_imbalance'], seed = params['seed'], strict = False)
            subsets = datasets['training']
            class_distributions = pd.DataFrame(compute_class_distributions(subsets, n_classes = n_classes_of[dataset]))
            class_distributions.index.name = 'subset'
            class_distributions.reset_index(inplace = True)
            subdata = class_distributions[class_distributions.index < 50]
            subdata.plot(kind = 'barh', x = 'subset', stacked = True, legend = None, ax = ax, width = 0.75)
            ax.set_axis_off()
            ax.set_title('class imbalance = {:.1f}'.format(0.0 if params['class_imbalance'] is None else params['class_imbalance']))

        fig.set_size_inches(9, 3.5)
        fig.tight_layout()

        fig.savefig(os.path.join(args.output, "{}-class-imbalance.pdf".format(dataset)), dpi = 300, bbox_inches = 'tight')

        plt.show()

    # display class distribution for each client for each dataset in an unbalanced setting
    grid_for_distributions = {
        'data_imbalance': [ 1 ],
        'class_imbalance': [ None, 1, 10 ],
        'seed': [ 0 ]
    }

    flattened_grid_for_distributions = list(dict(zip(grid_for_distributions, x)) for x in itertools.product(*grid_for_distributions.values()))

    for dataset in [ 'cifar10', 'cifar100', 'femnist' ]:
        fig, axes = plt.subplots(1, 3)

        for params, ax in zip(flattened_grid_for_distributions, numpy.ravel(axes)):
            datasets = loaders[dataset]('./data', heterogeneity_degree = params['class_imbalance'], samples_variance = params['data_imbalance'], seed = params['seed'], strict = False)
            subsets = datasets['training']
            class_distributions = pd.DataFrame(compute_class_distributions(subsets, n_classes = n_classes_of[dataset]))
            class_distributions.index.name = 'subset'
            class_distributions.reset_index(inplace = True)
            subdata = class_distributions[class_distributions.index < 50]
            subdata.plot(kind = 'barh', x = 'subset', stacked = True, legend = None, ax = ax, width = 0.75)
            ax.set_axis_off()
            ax.set_title('class imbalance = {:.1f}'.format(0.0 if params['class_imbalance'] is None else params['class_imbalance']))

        fig.set_size_inches(9, 3.5)
        fig.tight_layout()

        fig.savefig(os.path.join(args.output, "{}-class-imbalance-with-1-data-imbalance.pdf".format(dataset)), dpi = 300, bbox_inches = 'tight')

        plt.show()
import numpy
import os
import random
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from typing import Sequence, Union
import warnings

from torchvision.datasets import CIFAR10, CIFAR100

class FederatedSubset(Subset):
    '''
    This class represents the local subset held by each client in a federated simulation.
    '''

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)

        self.normalization = 255 if (isinstance(dataset, CIFAR10) or isinstance(dataset, CIFAR100)) else 1

    @property
    def data(self) -> torch.Tensor:
        if isinstance(self.dataset, TensorDataset):
            return self.dataset.tensors[0][self.indices]
        return self.dataset.data[self.indices]
    
    @property
    def targets(self) -> torch.Tensor:
        if isinstance(self.dataset, TensorDataset):
            return self.dataset.tensors[1][self.indices]
        if isinstance(self.dataset.targets, list):
            return torch.Tensor(self.dataset.targets)[self.indices]
        return self.dataset.targets[self.indices]
    
def set_seed(s: int):
    '''
    Sets the same random initialization seed across multiple libraries, and
    enables the usage of deterministic algorithms in PyTorch.

    Parameters
    ----------
    s: int
        Seed initialization value
    '''
    
    random.seed(s)
    numpy.random.seed(s)
    torch.manual_seed(s)
    
    if torch.version.cuda is not None and torch.version.cuda >= '10.2':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    else:
        torch.use_deterministic_algorithms(True)

def partition(
    dataset: Dataset,
    n_subsets: int,
    n_classes: int = None,
    heterogeneity_degree: float = 0.0,
    samples_variance: float = 0.0,
    return_indices: bool = False,
    seed: int = None,
    strict: bool = True
) -> Union[list[list[int]], list[FederatedSubset]]:
    '''
    Partitions a dataset in `n_subsets` heterogenously or homogenously according to
    `heterogeneity_degree` and `samples_variance`.

    Parameters
    ----------
    dataset: Dataset
        Dataset (pytorch)
    n_subsets: int
        Number of datasets for splitting
    n_classes: int
        Number of classes in the dataset, inferred if `None`
    heterogeneity_degree: float
        Class heterogeneity degree, by default is homogeneous
    samples_variance: float
        Standard deviation (%) in the number of samples for each client, `0` by default
    return_indices: bool
        To return indices instead of federated subsets
    seed: int
        Random seed initializer
    strict: bool
        In strict mode `heterogeneity_degree` and `samples_variance` are highly respected,
        otherwise flexibility is allowed

    Returns
    -------
    Union[list[list[int]], list[FederatedSubset]]
        If `return_indices` is `False` then returns federated subsets, otherwise indices

    Note
    ----
    Heterogeneity degree is the inverse of the concentration parameter of a Dirichelet distribution
    used to sample class ratios across each subset, whilst sample variance refers to the variance in
    the number of samples assigned to each subset, which is extracted from a log-normal distribution.
    '''

    # labels of dataset, to be computed
    labels: torch.Tensor = None
    # discriminate between kind of datasets
    if isinstance(dataset, TensorDataset):
        labels = dataset.tensors[1]
    else:
        assert hasattr(dataset, 'data'), "Dataset needs to have .data attribute"
        assert hasattr(dataset, 'targets'), "Dataset needs to have .targets attribute"
        labels = dataset.targets
    # number of classes extracted from data
    n_class_inferred = numpy.unique(labels).shape[0]
    # parameter checking
    assert 0 < n_subsets <= len(dataset), "Number of subsets must be between 1 and number of samples"
    assert heterogeneity_degree is None or heterogeneity_degree >= 0, "Dirichelet concentration must be a positive number"
    assert samples_variance is None or samples_variance >= 0, "Log-normal variance must be a positive number"
    assert n_classes is None or 0 < n_classes <= n_class_inferred, "Number of classes must be between 1 and number of labels' classes"
    # using less classes than expected is admissible even though it is signaled
    if n_classes < n_class_inferred:
        warnings.warn(f"Number of classes specified {n_classes} is inferior to inferred number of classes {n_class_inferred}")
    # random generator initialization for reproducibility
    if seed:
        set_seed(seed)
    # for each class extracts the indices of the corresponding samples 
    samples_indices = [ numpy.argwhere(numpy.array(labels) == k).reshape(-1) for k in range(n_classes) ]
    # number of samples in each subset of the partition (computed with respect to availability of samples)
    n_subset_samples = sum([ len(x) // n_subsets for x in samples_indices ])
    # no sample variance, so each user will receive same amount of samples
    if samples_variance is None or samples_variance == 0:
        n_subset_samples_sampled = numpy.repeat(n_subset_samples, n_subsets)
    # otherwise number of samples of each user are sampled according to lognormal distribution
    else:
        # log normal extraction
        n_subset_samples_sampled = numpy.random.lognormal(numpy.log(n_subset_samples), samples_variance, size = n_subsets)
        # normalization with respect to dataset size
        n_subset_samples_sampled = ((n_subset_samples_sampled / n_subset_samples_sampled.sum()) * n_subset_samples * n_subsets).astype(int)
    # in case of homogeneity each subset has the same fraction of data for each class
    if heterogeneity_degree is None or heterogeneity_degree == 0:
        n_partition_label_ratios_sampled = numpy.ones((n_subsets, n_classes)) / n_classes
    else:
        # otherwise for each class the ratio of corresponding data assigned are sampled according to dirichelet distribution
        alpha = 1.0 / heterogeneity_degree * numpy.ones(n_subsets)
        n_partition_label_ratios_sampled = numpy.random.dirichlet(alpha, size = n_classes).T
    # number of assigned samples used for each class
    n_consumed_class_samples = [ 0 for _ in range(n_classes) ]
    # indices of samples assigned to each subset
    subset_indices = [ [] for _ in range(n_subsets) ]
    # computation of number of samples that each client expects from each class (using floor)
    n_expected_samples_subset = numpy.floor(n_subset_samples_sampled.reshape(n_subsets, 1) * n_partition_label_ratios_sampled / n_partition_label_ratios_sampled.sum(axis = 1).reshape(n_subsets, 1)).astype(int)
    # number of samples effectively assigned to each client from each class
    n_effective_samples_subset = numpy.zeros((n_subsets, n_classes))
    # number of subsets (can be less that requested because some are empty, especially in strict mode)
    n_effective_subsets = None
    # shuffle data samples' indices of each class
    for k in range(n_classes):
        numpy.random.shuffle(samples_indices[k])
    # assigns samples to each subset
    for subset_index in range(n_subsets):
        # counter of number of samples assigned to each class for the current client
        n_received_samples = numpy.zeros(n_classes)
        # assigns samples of each class
        for k in range(n_classes):
            # in strict mode stops assigning and adding clients whenever the samples of any class are exhausted
            if strict and len(samples_indices[k]) - n_consumed_class_samples[k] < n_expected_samples_subset[subset_index, k]:
                n_effective_subsets = subset_index
                break
            # computes how many samples should be given to current client for current class, without exceeding the availability
            n_consumed_already = min(n_consumed_class_samples[k], len(samples_indices[k]))
            n_consumed_current = min(n_consumed_class_samples[k] + n_expected_samples_subset[subset_index, k], len(samples_indices[k]))
            # assigns samples to subset
            subset_indices[subset_index].extend(samples_indices[k][n_consumed_already:n_consumed_current])
            # updates counter
            n_received_samples[k] = max(n_consumed_current - n_consumed_already, 0)
            n_consumed_class_samples[k] = n_consumed_current
        # if any class availabilty is over then we stop building subsets and we exit
        if strict and n_effective_subsets is not None:
            subset_indices = subset_indices[:n_effective_subsets]
            break
        # updates counter of data assigned to subset
        n_effective_samples_subset[subset_index] = n_received_samples
    # indices of subsets
    indices = range(n_subsets)
    # this indicates how much importance is given to lack of data in subset when we need to account for
    # probabilities of distributing samples left from previous iterations
    inter_subset_samples_amount_importance = 0.25
    # distribution of left samples happens only in non-strict mode
    if not strict:
        for k in range(n_classes):
            # computes how many samples are left for each subset and for each class
            n_class_samples_left = len(samples_indices[k]) - n_consumed_class_samples[k]
            # computes how many samples are left for each subset
            n_samples_left = n_expected_samples_subset.sum(axis = 1) - n_effective_samples_subset.sum(axis = 1)
            n_samples_left[n_samples_left < 0] = 0
            # skip class samples distribution in case of no sample left
            if n_class_samples_left <= 0:
                continue
            # computes for each subset probability of getting new data of current class
            probabilities = 1 + (n_expected_samples_subset[:, k] - n_effective_samples_subset[:, k]) + inter_subset_samples_amount_importance * n_samples_left
            probabilities /= probabilities.sum()
            # weighted sampling with replacement a number of subsets equal to the number of samples left
            chosen_subsets = numpy.random.choice(
                indices,
                p = probabilities,
                size = n_class_samples_left,
                replace = True
            )
            # moves left samples to selected subsets for each subset
            for chosen_subset in chosen_subsets:
                subset_indices[chosen_subset].append(samples_indices[k][n_consumed_class_samples[k]])
                # updates number of samples assigned and consumed
                n_consumed_class_samples[k] += 1
                n_effective_samples_subset[chosen_subset, k] += 1
    # does not construct federated subset
    if return_indices:
        return subset_indices
    # returns federated subset, exclusively non empty
    return [ FederatedSubset(dataset, indices) for indices in subset_indices if len(indices) > 0 ]
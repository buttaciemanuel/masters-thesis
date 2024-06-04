from torchvision.datasets import CIFAR100
from typing import Any, Union

from . import utils


def cifar100(
    directory: str,
    n_subsets: int = 100,
    heterogeneity_degree: float = None,
    samples_variance: float = 0.0,
    transform: Any = None,
    target_transform: Any = None,
    seed: int = None,
    strict: bool = True
) -> dict[str, list[utils.FederatedSubset]]:
    '''
    Loads the `CIFAR100` dataset and partitions it into `n_subsets` training subsets and 
    `n_subsets` testing subsets according to heterogeneity parameters `heterogeneity_degree` 
    and `samples_variance`.

    Parameters
    ----------
    directory: str
        Root directory to load the dataset
    n_subsets: int
        Number of datasets for splitting
    heterogeneity_degree: float
        Class heterogeneity degree, by default is homogeneous
    samples_variance: float
        Standard deviation (%) in the number of samples for each client, `0` by default
    transform: Any
        Transformation to apply on data samples, `None` by default
    target_transform: Any
       Transformation to apply on data labels, `None` by default
    seed: int
        Random seed initializer
    strict: bool
        In strict mode `heterogeneity_degree` and `samples_variance` are highly respected,
        otherwise flexibility is allowed

    Returns
    -------
    dict[str, list[utils.FederatedSubset]]
        Returns the lists of subsets of training clients and testing clients
    '''

    training = CIFAR100(
        root = directory, 
        train = True, 
        download = True,
        transform = transform,
        target_transform = target_transform
    )

    testing = CIFAR100(
        root = directory, 
        train = False, 
        download = True,
        transform = transform,
        target_transform = target_transform
    )

    return {
        'training': utils.partition(
            training,
            n_subsets = min(len(training), n_subsets),
            n_classes = 100,
            heterogeneity_degree = heterogeneity_degree,
            samples_variance = samples_variance,
            return_indices = False,
            seed = seed,
            strict = strict
        ),
        'testing': utils.partition(
            testing,
            n_subsets = min(len(testing), n_subsets),
            n_classes = 100,
            heterogeneity_degree = heterogeneity_degree,
            samples_variance = samples_variance,
            return_indices = False,
            seed = seed,
            strict = strict
        )
    }
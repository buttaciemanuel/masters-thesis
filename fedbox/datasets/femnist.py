import os.path as path
import torch
from torch.utils.data import Dataset
from typing import Any, Tuple, Union

from . import utils


class FEMNIST(Dataset):
    '''
    This class loads the FEMNIST dataset, specifically only its 10 classes subset with written digit in [0, 9].

    Note
    ----
    The dataset must be downloaded beforehand using the instructions in the file README.md.
    '''

    def __init__(
        self,
        root: str, 
        train: bool = True,
        transform: Any = None,
        target_transform: Any = None, 
        download = True
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data = None
        self.targets = None

        directory = path.join(self.root, 'femnist')
        # NOTE: dataset must be downloaded first by following the instructions in README.md file
        if path.exists(directory):
            self.data, self.targets, _ = torch.load(path.join(directory, 'training.pt' if self.train else 'testing.pt'))
            self.targets = self.targets.to(dtype = torch.float32)
        else:
            raise RuntimeError('FEMNIST dataset not found in directory {}. It must be downloaded first using the README.md file.'.format(root))

        self.data = self.data.unsqueeze(3)
        self.targets = self.targets.to(dtype = torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x, y = self.data[index], self.targets[index]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y
    
    def __len__(self) -> int:
        return self.data.shape[0]

def femnist(
    directory: str,
    n_subsets: int = 1000,
    heterogeneity_degree: float = None,
    samples_variance: float = 0.0,
    transform: Any = None,
    target_transform: Any = None,
    seed: int = None,
    strict: bool = True
) -> dict[str, list[utils.FederatedSubset]]:
    '''
    Loads the `FEMNIST` dataset and partitions it into `n_subsets` training subsets and 
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

    training = FEMNIST(
        root = directory, 
        train = True, 
        download = True,
        transform = transform,
        target_transform = target_transform
    )

    testing = FEMNIST(
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
            n_classes = 10,
            heterogeneity_degree = heterogeneity_degree,
            samples_variance = samples_variance,
            return_indices = False,
            seed = seed,
            strict = strict
        ),
        'testing': utils.partition(
            testing,
            n_subsets = min(len(testing), n_subsets),
            n_classes = 10,
            heterogeneity_degree = heterogeneity_degree,
            samples_variance = samples_variance,
            return_indices = False,
            seed = seed,
            strict = strict
        )
    }
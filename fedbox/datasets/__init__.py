'''
This submodule defines the federated version of the following datasets.
<ol>
  <li>FEMNIST (written digits subsets) (see <a href="https://arxiv.org/pdf/1812.01097">paper</a>) that can be loaded with <b>femnist(...)</b>.</li>
  <li>CIFAR10 and CIFAR100 (see <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">paper</a>) that can be loaded with <b>cifar10(...)</b> and <b>cifar100(...)</b>, respectively.</li>
</ol>
In addition, here we define the routine **partition(...)** that partitions a centralized dataset in multiple subsets where each corresponds to a client. This function
is responsible injecting the specified level of heterogeneity in the generation of the federated subsets.
'''

from .cifar10 import cifar10
from .cifar100 import cifar100
from .femnist import femnist, FEMNIST
from .utils import FederatedSubset, partition
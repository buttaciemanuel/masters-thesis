'''
This package allows running customizable federated simulations on two different dataset families.
<ol>
  <li>FEMNIST (written digits subsets) (see <a href="https://arxiv.org/pdf/1812.01097">paper</a>)</li>
  <li>CIFAR10 and CIFAR100 (see <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">paper</a>)</li>
</ol>

Three possible algorithms can be chosen.
<ol>
  <li>FedAvg (see <a href="https://arxiv.org/pdf/1602.05629">paper</a>)</li>
  <li>FedProx (see <a href="https://arxiv.org/pdf/1812.06127">paper</a>)</li>
  <li>Our novel framework proposed in the paper alongside this supplemental code
</ol>

Examples
--------
This example lets you run a simulation on imbalanced FEMNIST dataset using our proposed algorithm
with beta equal to 0.70 (perturbation parameter) and a logistic regression model. Moreover, the simulation 
in scheduled with 200 global rounds and 10 local optimization epochs per each. The number of training
and testing clients is 100 (per each group). The simulation saves the data in the './simulations'
directory as a JSON file.

>>> n_rounds = 200
>>> n_epochs = 10
>>> class_imbalance = 10
>>> data_imbalance = 1
>>> n_clients = 100
>>> datasets = femnist(
        './data', 
        n_subsets = n_clients,
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ]),
        heterogeneity_degree = class_imbalance,
        samples_variance = data_imbalance,
        strict = False,
        seed = 0
    )
>>> model = MultinomialLogisticRegression(n_inputs = 784, n_classes = 10)
>>> scheme = AdjancencyWeightingScheme(datasets)
>>> logger = Logger(
        directory = './simulations', 
        simulation = { 'dataset': 'femnist', 'class_imbalance': class_imbalance, 'data_imbalance': data_imbalance }, 
        convergence = lambda values: values['step'] if values['accuracy.testing'] >= 0.75 else None, 
        enable = True
    )
>>> coordinator = ours.Coordinator(beta = 0.70, model, datasets, scheme, logger)
>>> coordinator.run(n_iterations = n_rounds, n_epochs = n_epochs, batch_size = 256, step_size = 1e-3, l2_penalty = 1e-4)
>>> logger.flush()
'''
import torch
from torch import nn


class MultinomialLogisticRegression(nn.Module):
    '''
    This class implements the multinomial logistic regression
    for multiclass classification. The combination of this model
    and loss criterion yields a strongly convex loss objective.

    Note
    ----
    A cross-entropy loss is utilized during optimization, in Pytorch
    it is equivalent to log-softmax and negative log likelihood. Thus,
    we do not need to apply softmax on the output layer, see [link](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
    '''

    def __init__(self, n_inputs: int, n_classes: int):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(n_inputs, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.linear(x)
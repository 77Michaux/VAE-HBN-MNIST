import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.GELU(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

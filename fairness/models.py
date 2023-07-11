from django.db import models

import torch

# Create your models here.
class PredictorModel(torch.nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(104, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

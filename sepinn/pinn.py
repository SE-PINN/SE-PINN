#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from sepinn.hublayer import HubLayer

# Optional Hardware Acceleration
if torch.cuda.is_available():  # Use T4 GPU on Google Colab
    torch.cuda.init()
    torch.cuda.is_initialized()
    torch.set_default_tensor_type(
        "torch.cuda.FloatTensor"
    )  # torch.set_default_dtype() and torch.set_default_device()
    device = "cuda"  # torch.device("cuda")
else:
    device = "cpu"

# !pip install torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()

torch.manual_seed(0)  # Specify the random seed for reproducibility.


# This is a shortcut to plot pytorch tensors (they need to be in numpy form for matplotlib).
def to_plot(x):
    return x.detach().cpu().numpy()


class PINN(nn.Module):
    """
    Physics-informed neural network (PINN) to solve the Schrodinger equation.

    Attributes
    ----------
    x0 : float
        the spatial position of the leftmost point of the quantum-mechanical potential
    xN : float
        the spatial position of the rightmost point of the quantum-mechanical potential
    dx : float
        the uniform spatial Euclidean distance between adjacent points
    N : int
        the count of points
    activation : builtin_function_or_method
        the activation function
    sym : int
        whether to enforce even symmetry (1) or odd symmetry (-1) or not to enforce symmetry (0)

    Methods
    -------
    swap_symmetry
        Swap the symmetry of the prediction of the model between even symmetry and odd symmetry.

    forward(x)
        Forward pass.
    """

    def __init__(self, grid_params, activation, sym=0):
        super(PINN, self).__init__()

        self.x0, self.xN, self.dx, self.N = grid_params
        self.activation = activation
        self.sym = sym

        # Architecture of the Model

        self.energy_node = nn.Linear(1, 1)
        self.fc1_bypass = nn.Linear(1, 50)
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)

        # Automatic detection of whether to enforce even symmetry or odd symmetry via a hub layer
        if sym == 1:
            self.output = HubLayer(50, 1, 1, 0)  # Even Symmetry
        elif sym == -1:
            self.output = HubLayer(50, 1, 0, 1)  # Odd Symmetry
        else:
            self.output = nn.Linear(50, 1)

    def swap_symmetry(self):
        if self.sym == 0:
            print("Tried to swap symmetry although none is enforced.")
            return
        self.output.flip_sym()

    def forward(self, x):
        # lambda layer for energy
        energy = self.energy_node(torch.ones_like(x))

        N = torch.cat((x, energy), 1)
        N = self.activation(self.fc1(N))
        N = self.activation(self.fc2(N))
        N = self.output(N)  # where symmetrization occurs if enforced

        wf = N

        return wf, energy

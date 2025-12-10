#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from sepinn.hub_layer import HubLayer

__all__ = ["BasePINN"]


# This is a shortcut to plot pytorch tensors (they need to be in numpy form for matplotlib).
def to_plot(x):
    return x.detach().cpu().numpy()


class BasePINN(nn.Module):
    """
    A base class for a physics-informed neural network (PINN) for
    solving the Schrodinger equation.

    Attributes
    ----------
    x0 : float
        The spatial position of the leftmost point of the
        quantum-mechanical potential.
    xN : float
        The spatial position of the rightmost point of the
        quantum-mechanical potential.
    dx : float
        The uniform spatial Euclidean distance between adjacent points.
    N : int
        The count of points of the quantum-mechanical potential.
    activation : builtin_function_or_method
        The activation function.
    sym : int
        Whether to enforce even symmetry (1) or odd symmetry (-1) or not
        to enforce symmetry (0).

    Methods
    -------
    swap_symmetry
        Swap the symmetry of the prediction of the model between even
        symmetry and odd symmetry.

    forward(x)
        Forward pass.
    """

    def __init__(self, grid_params, activation, sym=0):
        super(BasePINN, self).__init__()

        self.x0, self.xN, self.dx, self.N = grid_params
        self.activation = activation
        self.sym = sym

        # Architecture of the Model

        self.energy_node = nn.Linear(1, 1)

        self.fc1_bypass = nn.Linear(1, 50)
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)

        # Selection of the Output Layer
        if sym == 1:
            # Enforcement of Even Symmetry
            self.output_layer = HubLayer(50, 1, 1, 0)
        elif sym == -1:
            # Enforcement of Odd Symmetry
            self.output_layer = HubLayer(50, 1, 0, 1)
        else:
            # No Enforcement of Symmetry
            self.output_layer = nn.Linear(50, 1)

    def swap_symmetry(self):
        if isinstance(self.output_layer, HubLayer):
            self.output_layer.flip_sym()
        else:
            print("Symmetry cannot be swapped because it is not enforced.")

    def forward(self, x):
        # Lambda Layer for Energy
        energy = self.energy_node(torch.ones_like(x))

        N = torch.cat((x, energy), dim=1)
        N = self.activation(self.fc1(N))
        N = self.activation(self.fc2(N))
        wf = self.output_layer(N)  # Possible enforcement of symmetry.

        return wf, energy

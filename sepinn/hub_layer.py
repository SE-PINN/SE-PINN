#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class HubLayer(nn.Module):
    """
    A hub layer, which is used to constrain the prediction of the model
    to respect even symmetry (symmetry about f(x) = 0) or odd symmetry
    (symmetry about f(x) = x). The mathematical basis is presented at
    https://arxiv.org/pdf/1904.08991.pdf. The constructor is adapted
    from https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77.

    Attributes
    ----------
    size_in : int
        The length of the input of the layer.
    size_out : int
        The length of the output of the layer.
    weights : torch.nn.parameter.Parameter
        The weights of the layer.
    bias : torch.nn.parameter.Parameter
        The bias of the layer.
    even : int
        1 to enforce even symmetry.
    odd : int
        -1 to enforce odd symmetry.

    Methods
    -------
    flip_sym
        Swap the symmetry between even symmetry and odd symmetry.

    forward(x)
        Forward pass.
    """

    def __init__(self, size_in, size_out, even, odd):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out

        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)

        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        self.even = even
        self.odd = odd

        # Initialization of Weights (Kaiming Initialization)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

        # Initialization of Biases (LeCun Initialization)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def flip_sym(self):
        self.even = 1 - self.even
        self.odd = 1 - self.odd
        return

    def forward(self, x):
        h_plus = x  # This is x(t).
        h_minus = torch.flip(x, [0])  # This is x(-t).

        H_plus = h_plus + h_minus
        H_minus = h_plus - h_minus

        N = (self.even * (1 / 2) * torch.mm(H_plus, self.weights.t())) + (
            self.odd * (1 / 2) * torch.mm(H_minus, self.weights.t())
        )

        return N

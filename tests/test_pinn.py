import pytest
import torch
import torch.nn as nn

from sepinn.pinn import PINN

N = 500
x0, xN = -5.0, 5.0
dx = (xN - x0) / N
grid_params = x0, xN, dx, N

x = torch.linspace(x0, xN, N - 1).view(-1, 1)
k = 100
V = 0.5 * k * x**2


class TestInitialization:
    def test_class(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert isinstance(model, PINN)

    def test_grid_params(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert model.x0 == grid_params[0]

    def test_activation(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert model.activation == torch.tanh

    def test_sym(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert model.sym == 0

    def test_energy_node(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert isinstance(model.energy_node, nn.Linear)

    def test_fc1_bypass(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert isinstance(model.fc1_bypass, nn.Linear)

    def test_fc1(self):
        model = PINN(grid_params, activation=torch.tanh)

        assert isinstance(model.fc1, nn.Linear)


def test_swap_symmetry():
    model = PINN(grid_params, activation=torch.tanh)

    model.swap_symmetry()

    assert model.sym == 0

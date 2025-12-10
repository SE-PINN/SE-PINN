import pytest
import torch
import torch.nn as nn

from sepinn.base_pinn import BasePINN

N = 500
x0, xN = -5.0, 5.0
dx = (xN - x0) / N
grid_params = x0, xN, dx, N

x = torch.linspace(x0, xN, N - 1).view(-1, 1)
k = 100
V = 0.5 * k * x**2


class TestInitialization:
    def test_class(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert isinstance(base_pinn, BasePINN)

    def test_grid_params(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert base_pinn.x0 == grid_params[0]

    def test_activation(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert base_pinn.activation == torch.tanh

    def test_sym(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert base_pinn.sym == 0

    def test_energy_node(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert isinstance(base_pinn.energy_node, nn.Linear)

    def test_fc1_bypass(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert isinstance(base_pinn.fc1_bypass, nn.Linear)

    def test_fc1(self):
        base_pinn = BasePINN(grid_params, activation=torch.tanh)

        assert isinstance(base_pinn.fc1, nn.Linear)


def test_swap_symmetry():
    base_pinn = BasePINN(grid_params, activation=torch.tanh)

    base_pinn.swap_symmetry()

    assert base_pinn.sym == 0

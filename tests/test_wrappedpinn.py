import pytest
import torch

from sepinn.wrappedpinn import WrappedPINN

N = 500
x0, xN = -5.0, 5.0
dx = (xN - x0) / N
grid_params = x0, xN, dx, N

x = torch.linspace(x0, xN, N - 1).view(-1, 1)
k = 100
V = 0.5 * k * x**2

params = {
    "grid_params": grid_params,
    "activation": torch.tanh,
    "potential": V,
    "sym": 1,
}


class TestInitialization:
    def test_V(self):
        model = WrappedPINN(**params)

        assert (model.V == V).all()

    def test_cur_loss(self):
        model = WrappedPINN(**params)

        assert model.cur_loss == 0

    def test_cur_energy(self):
        model = WrappedPINN(**params)

        assert model.cur_energy == 0

    def test_cur_wf(self):
        model = WrappedPINN(**params)

        assert model.cur_wf == 0

    def test_losses(self):
        model = WrappedPINN(**params)

        assert model.losses == []

    def test_energies(self):
        model = WrappedPINN(**params)

        assert model.energies == []

    def test_wfs(self):
        model = WrappedPINN(**params)

        assert model.wfs == []

    def test_basis(self):
        model = WrappedPINN(**params)

        assert model.basis == []

    def test_basis_sum(self):
        model = WrappedPINN(**params)

        assert isinstance(model.basis_sum, torch.Tensor)


def test_init_optimizer():
    model = WrappedPINN(**params)

    model.init_optimizer(optim="LBFGS")

    assert model.opt_name == "LBFGS"

    model.init_optimizer(optim="Adam")

    assert model.opt_name == "Adam"


def test_change_lr():
    model = WrappedPINN(**params)
    model.init_optimizer(optim="LBFGS")

    for lr in range(1, 1000):
        lr = 1 / lr
        model.change_lr(lr)
        assert model.opt.param_groups[0]["lr"] == lr

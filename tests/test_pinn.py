import torch

from sepinn.pinn import PINN

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
        pinn = PINN(**params)

        assert (pinn.V == V).all()

    def test_cur_loss(self):
        pinn = PINN(**params)

        assert pinn.cur_loss == 0

    def test_cur_energy(self):
        pinn = PINN(**params)

        assert pinn.cur_energy == 0

    def test_cur_wf(self):
        pinn = PINN(**params)

        assert isinstance(pinn.cur_wf, torch.Tensor)
        assert pinn.cur_wf.shape == x.shape
        assert (pinn.cur_wf == 0).all()

    def test_losses(self):
        pinn = PINN(**params)

        assert pinn.losses == []

    def test_energies(self):
        pinn = PINN(**params)

        assert pinn.energies == []

    def test_wfs(self):
        pinn = PINN(**params)

        assert pinn.wfs == []

    def test_basis(self):
        pinn = PINN(**params)

        assert pinn.basis == []

    def test_basis_sum(self):
        pinn = PINN(**params)

        assert isinstance(pinn.basis_sum, torch.Tensor)


def test_init_optimizer():
    pinn = PINN(**params)

    pinn.init_optimizer(optimizer_name="LBFGS")

    assert pinn.optimizer_name == "LBFGS"

    pinn.init_optimizer(optimizer_name="Adam")

    assert pinn.optimizer_name == "Adam"


def test_change_lr():
    pinn = PINN(**params)
    pinn.init_optimizer(optimizer_name="LBFGS")

    for lr in range(1, 1000):
        lr = 1 / lr
        pinn.change_lr(lr)
        assert pinn.optimizer.param_groups[0]["lr"] == lr

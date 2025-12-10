#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from sepinn.pinn import PINN

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


# This is a shortcut to plot pytorch tensors (they need to be in numpy form for matplotlib).
def to_plot(x):
    return x.detach().cpu().numpy()


class WrappedPINN:
    def __init__(self, grid_params, activation, potential, sym):
        self.x0, self.xN, self.dx, self.N = grid_params

        self.x = torch.linspace(self.x0, self.xN, self.N - 1).view(-1, 1)
        self.V = potential

        self.model = PINN(grid_params, activation, sym)
        self.model.to(device)

        # Persistent information about the predicted basis.
        self.basis = []
        self.basis_sum = torch.zeros_like(self.x)

        # Current output of the model.
        self.cur_loss = 0
        self.cur_energy = 0
        self.cur_wf = 0

        # Lists for visualization of the history of the model.
        self.losses = []
        self.energies = []
        self.wfs = []

    def init_optimizer(self, optim, lr=1e-2):
        if optim == "LBFGS":
            self.opt = torch.optim.LBFGS(self.model.parameters(), lr=lr)
        elif optim == "Adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            print("Invalid Optimizer")
        self.opt_name = optim

    def change_lr(self, lr):
        """
        On-the-fly (runtime) control of the learning rate.
        """
        self.opt.param_groups[0]["lr"] = lr

    def swap_symmetry(self):
        """
        On-the-fly (runtime) control of the enforced symmetry.
        """
        self.model.swap_symmetry()

    def add_to_basis(self, base=None):
        if base is None:
            base = self.cur_wf.clone().detach()
        self.basis.append(base)
        self.basis_sum += base

    def closure(self):
        self.opt.zero_grad()
        loss_value = self.loss(self.x)
        loss_value.backward()
        return loss_value

    def loss(self, x):
        self.x.requires_grad = True
        wf, energy = self.model(self.x)

        d = torch.autograd.grad(wf.sum(), x, create_graph=True)[0]
        dd = torch.autograd.grad(d.sum(), x, create_graph=True)[0]  # 2nd derivative
        SE_loss = torch.sum((-0.5 * dd + self.V * wf - energy * wf) ** 2) / self.N

        NL_loss = (torch.sum(wf**2) - 1 / self.dx) ** 2

        Orth_loss = (torch.sum(wf * self.basis_sum) * self.dx) ** 2

        loss = (
            SE_loss + NL_loss + Orth_loss + 0.5 * (wf[0] ** 2 + wf[-1] ** 2)
        )  # boundary loss

        self.cur_wf, self.cur_energy, self.cur_loss = wf, energy[0].item(), loss.item()

        return loss

    def train(self, epochs):
        for i in range(epochs):
            if self.opt_name == "LBFGS":
                loss = self.opt.step(self.closure)
                if loss.item() == torch.nan:
                    print("NAN loss")
                    break

            if self.opt_name == "Adam":
                self.opt.zero_grad()
                loss = self.loss(self.x)
                loss.backward()
                self.opt.step()

            self.wfs.append(self.cur_wf)
            self.energies.append(self.cur_energy)
            self.losses.append(self.cur_loss)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss during Training")
        plt.show()

    def plot_energy(self):
        plt.plot(self.energies)
        plt.title("Predicted Energy Eigenvalue during Training")
        plt.xlabel("Epoch")
        plt.ylabel("Energy Eigenvalue")
        plt.show()

    def plot_wf(self, idx=None, ref=None):
        # fig = plt.figure(figsize=(6,4))
        if idx is None:
            psi = self.cur_wf
        else:
            psi = self.wfs[idx]
        plt.plot(to_plot(self.x), to_plot(psi), "r-", linewidth=2, label="Prediction")
        plt.plot(
            to_plot(self.x), -to_plot(psi), "b-", linewidth=2, label="- Prediction"
        )
        if ref is not None:
            plt.plot(to_plot(self.x), ref, "k--", linewidth=2, label="Ground Truth")

        plt.xlabel("x")
        plt.ylabel("Energy Eigenvector")
        plt.title(
            f"Predicted Energy Eigenvector (E = {self.cur_energy:.2f}, norm = {torch.sum(psi**2) * self.dx:.2f})"
        )
        plt.legend()
        plt.rcParams["figure.figsize"] = (6, 4)
        plt.show()

    def create_gif(self, name, ref_wf=None, ref_ener=None, epoch_range=None):
        if epoch_range is None:
            epoch_range = 0, len(self.losses)
        num_frames = epoch_range[1] - epoch_range[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        def animate(i):
            idx = epoch_range[0] + i
            ax = axes[0]
            ax.clear()
            ax.plot(
                to_plot(self.x),
                to_plot(self.wfs[idx]),
                "r-",
                linewidth=2,
                label="Prediction",
            )
            ax.plot(
                to_plot(self.x),
                -to_plot(self.wfs[idx]),
                "b-",
                linewidth=2,
                label="- Prediction",
            )
            if ref_wf is not None:
                ax.plot(
                    to_plot(self.x), ref_wf, "k--", linewidth=2, label="Ground Truth"
                )
            ax.set_title(f"Epoch {i}: Energy = {self.energies[idx]:.2f}")
            ax.legend()
            ax.set_ylim([-1.5, 1.5])

            ax = axes[1]
            ax.clear()
            ax.plot(np.arange(epoch_range[0], idx), self.energies[epoch_range[0] : idx])
            ax.set_xlim([epoch_range[0], epoch_range[1]])
            if ref_ener is not None:
                ax.axhline(ref_ener, color="k", linestyle="--", label="Ground Truth")
            ax.set_title("Energy")
            ax.legend()

        ani = FuncAnimation(fig, animate, frames=num_frames - 1, interval=500)
        ani.save(name + ".gif", dpi=100, writer=PillowWriter(fps=50))
        plt.close()

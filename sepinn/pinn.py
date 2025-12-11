#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys

try:
    import IPython
    import IPython.display
    import matplotlib_inline

    matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
    IN_NOTEBOOK = True
except (ImportError, NotImplementedError):
    IN_NOTEBOOK = False

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    track,
)

from sepinn.base_pinn import BasePINN

__all__ = ["PINN"]

plt.rcParams["figure.figsize"] = (6.4, 4.8)


# Optional Hardware Acceleration
# Use Runtime > Change runtime type > T4 GPU on Google Colab.
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.is_initialized()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print(f"Using {device}.")

# Settings for Reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"  # cuBLAS
np.random.seed(seed)
random.seed(seed)


# Convenience Function for Plotting with PyTorch Tensors
def _to_plot(x):
    return x.detach().cpu().numpy()


class PINN:
    """
    An implementation of a physics-informed neural network (PINN) for
    solving the Schrodinger equation with infrastructure for training
    and visualization.

    Attributes
    ----------
    x : torch.Tensor
        The numerical grid of the physical system.
    x0 : float
        The leftmost point of the numerical grid.
    xN : float
        The rightmost point of the numerical grid.
    N : int
        The count of points that the numerical grid has.
    V : torch.Tensor
        The potential.
    basis : list
        The basis.
    basis_sum : torch.Tensor
        The sum of the eigenvectors of the basis.
    cur_loss : float
        The current loss of the model.
    cur_energy : float
        The current prediction of the energy eigenvalue from the model.
    cur_wf : torch.Tensor
        The current prediction of the energy eigenvector from the model.
    losses : list
        A list of all losses of the model.
    energies : list
        A list of all energy eigenvalues of the model.
    wfs : list
        A list of all energy eigenvectors of the model.

    Methods
    -------
    init_optimizer(optimizer_name='LBFGS', lr=1e-3)
        Initialize the optimizer for training the model.
    change_lr(lr)
        Change the learning rate for training the model.
    swap_symmetry
        Swap the symmetry of the model.
    add_to_basis(base=None)
        Add the predicted energy eigenvector to the basis.
    closure
        Necessary for computing the loss with the L-BFGS optimizer.
    loss_fn(x)
        Computes the loss of the model.
    train(epochs=10)
        Trains the model.
    plot(metrics=['loss', 'energy', 'wf'], ref_energy=None, ref_wf=None)
        Plot a set of metrics.
    plot_loss
        Plot the loss.
    plot_energy(ref_energy=None)
        Plot the energy eigenvalue that is predicted by the model.
    plot_wf(idx=None, ref_wf=None)
        Plot the energy eigenvector that is predicted by the model.
    animate(filename, ref_energy=None, ref_wf=None, epoch_range=None,
            display=False)
        Plot the predictions of the model as an animation.
    """

    def __init__(self, grid_params, activation, potential, sym):
        self.x0, self.xN, self.dx, self.N = grid_params
        self.x = torch.linspace(self.x0, self.xN, self.N - 1).view(-1, 1)
        self.V = potential

        self.model = BasePINN(grid_params, activation, sym)
        self.model.to(device)

        # Persistent information about the predicted basis.
        self.basis = []
        self.basis_sum = torch.zeros_like(self.x)

        # Current values of metrics.
        self.cur_loss = 0
        self.cur_energy = 0
        self.cur_wf = torch.zeros_like(self.x)

        # All values of metrics.
        self.losses = []
        self.energies = []
        self.wfs = []

    def init_optimizer(self, optimizer_name="LBFGS", lr=1e-3):
        if optimizer_name == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            print("The name of the optimizer is invalid.")
            return

        self.optimizer_name = optimizer_name

    def change_lr(self, lr):
        """
        On-the-fly (runtime) control of the learning rate.
        """

        self.optimizer.param_groups[0]["lr"] = lr

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
        """
        The closure method is necessary for the L-BFGS optimizer since
        it evaluates the loss of the model at multiple points in
        parameter space at each step of training in contrast with the
        other optimizers in PyTorch.
        """

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.x)
        loss.backward()
        return loss

    def loss_fn(self, x):
        self.x.requires_grad = True

        wf, energy = self.model(self.x)

        # First Derivative
        d = torch.autograd.grad(wf.sum(), x, create_graph=True)[0]
        # Second Derivative
        dd = torch.autograd.grad(d.sum(), x, create_graph=True)[0]

        # SE Loss
        SE_loss = torch.sum((-0.5 * dd + self.V * wf - energy * wf) ** 2)
        SE_loss /= self.N

        # Normality Loss
        normality_loss = (torch.sum(wf**2) - 1 / self.dx) ** 2

        # Orthogonality Loss
        orthogonality_loss = (torch.sum(wf * self.basis_sum) * self.dx) ** 2

        # Boundary Loss
        boundary_loss = 0.5 * (wf[0] ** 2 + wf[-1] ** 2)

        # Total Loss
        loss = SE_loss + normality_loss + orthogonality_loss + boundary_loss

        self.cur_wf = wf
        self.cur_energy = energy[0].item()
        self.cur_loss = loss.item()

        return loss

    def train(self, epochs=10):
        for _ in track(range(epochs), description="Training... "):
            if isinstance(self.optimizer, torch.optim.LBFGS):
                loss = self.optimizer.step(self.closure)

                if loss.item() == torch.nan:
                    print("The loss is NAN.")
                    break
            elif isinstance(self.optimizer, torch.optim.Adam):
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.x)
                loss.backward()
                self.optimizer.step()

            self.wfs.append(self.cur_wf)
            self.energies.append(self.cur_energy)
            self.losses.append(self.cur_loss)

    def plot(self, metrics=["loss", "energy", "wf"], ref_energy=None, ref_wf=None):
        def route(metric):
            if metric == "loss":
                self.plot_loss()
            elif metric == "energy":
                self.plot_energy(ref_energy=ref_energy)
            elif metric == "wf":
                self.plot_wf(ref_wf=ref_wf)
            else:
                message = "The metric must be 'loss', 'energy', or 'wf' "
                message += f"rather than {repr(metric)}."
                print(message)

        if isinstance(metrics, str):
            route(metrics)
        elif isinstance(metrics, list):
            for metric in metrics:
                route(metric)
        else:
            message = f"The type of the metrics parameter must be {repr(str)} "
            message += f"or {repr(list)} rather than {type(metrics)}."
            print(message)

    def plot_loss(self):
        _ = plt.figure(figsize=(6.4, 4.8))
        plt.plot(self.losses)
        plt.yscale("log")
        plt.title("Loss during Training", loc="left")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.subplots_adjust(left=0.2, right=0.95)
        if len(self.losses) < 10:
            plt.xticks(range(len(self.losses)))
        plt.grid(alpha=0.2, which="both")
        plt.show()
        plt.close()

    def plot_energy(self, ref_energy=None):
        _ = plt.figure(figsize=(6.4, 4.8))
        plt.plot(self.energies)
        if ref_energy is not None:
            plt.axhline(ref_energy, color="k", linestyle="--", label="Ground Truth")
        plt.title("Energy Eigenvalue during Training", loc="left")
        plt.xlabel("Epoch")
        plt.ylabel("Energy Eigenvalue")
        plt.subplots_adjust(left=0.15, right=0.95)
        if len(self.energies) < 10:
            plt.xticks(range(len(self.energies)))
        plt.grid(alpha=0.2)
        plt.show()
        plt.close()

    def plot_wf(self, idx=None, ref_wf=None):
        _ = plt.figure(figsize=(6.4, 4.8))

        if idx is None:
            psi = self.cur_wf.detach()
            energy = self.cur_energy
        else:
            psi = self.wfs[idx].detach()
            energy = self.energies[idx]
        norm = torch.sum(psi**2) * self.dx

        plt.plot(_to_plot(self.x), _to_plot(psi), "r-", label="Prediction")
        plt.plot(_to_plot(self.x), -_to_plot(psi), "b-", label="- Prediction")

        if ref_wf is not None:
            plt.plot(_to_plot(self.x), ref_wf, "k--", label="Ground Truth")

        title = f"Energy Eigenvector (Norm of {norm:.2f} and Energy of "
        title += f"{energy:.2f})"

        plt.title(title, loc="left")
        plt.xlabel("Position")
        plt.ylabel("Probability Amplitude")
        plt.subplots_adjust(left=0.15, right=0.95)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()
        plt.close()

    def animate(
        self, filename, ref_wf=None, ref_energy=None, epoch_range=None, display=False
    ):
        # Use rich.progress.Progress to display progress.
        column_list = [
            TextColumn("Animating..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        ]
        with Progress(*column_list) as progress:
            if epoch_range is None:
                epoch_range = (0, len(self.losses))
            num_frames = epoch_range[1] - epoch_range[0]

            task = progress.add_task("Animating...", total=num_frames)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Function for FuncAnimation in Matplotlib
            def plot_frame(i):
                progress.update(task, advance=1)

                idx = epoch_range[0] + i

                psi = self.wfs[idx]
                norm = torch.sum(psi**2) * self.dx

                # Plot Energy Eigenvector
                ax = axes[0]
                ax.clear()
                ax.plot(
                    _to_plot(self.x), _to_plot(self.wfs[idx]), "r-", label="Prediction"
                )
                ax.plot(
                    _to_plot(self.x),
                    -_to_plot(self.wfs[idx]),
                    "b-",
                    label="- Prediction",
                )
                if ref_wf is not None:
                    ax.plot(_to_plot(self.x), ref_wf, "k--", label="Ground Truth")
                ax.set_title(f"Energy Eigenvector: Norm of {norm:.2f}", loc="left")
                ax.set_xlabel("Position")
                ax.set_ylabel("Probability Amplitude")
                ax.set_ylim([-1.5, 1.5])
                ax.grid(alpha=0.2)
                ax.legend()

                # Plot Energy Eigenvalue
                ax = axes[1]
                ax.clear()
                ax.plot(
                    np.arange(epoch_range[0], idx + 1),
                    self.energies[epoch_range[0] : idx + 1],
                )
                if ref_energy is not None:
                    ax.axhline(
                        ref_energy, color="k", linestyle="--", label="Ground Truth"
                    )
                ax.set_title(f"Energy Eigenvalue: {self.energies[idx]:.2f}", loc="left")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Energy Eigenvalue")
                ax.set_xlim([epoch_range[0], epoch_range[1]])
                ax.grid(alpha=0.2)
                ax.legend()

                return []

            ani = FuncAnimation(fig, plot_frame, frames=num_frames - 1, interval=300)
            ani.save(filename + ".gif", dpi=200, writer=PillowWriter(fps=50))
            plt.close()

        if display:
            if IN_NOTEBOOK:
                if "google.colab" in sys.modules:
                    filename = "/content/" + filename + ".gif"
                else:
                    filename = filename + ".gif"

                IPython.display.display(IPython.display.Image(filename=filename))
            else:
                print(f"Animation saved to {filename}.gif")

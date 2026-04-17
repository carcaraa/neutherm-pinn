"""
Training data generation, loss functions, and training pipelines.

Submodules
----------
dataset : Parametric sweep generation via Latin Hypercube Sampling.
losses : Loss functions (MSE, relative L2) for surrogate and PINN.
train_surrogate : Training pipeline for the data-driven surrogate model.
"""

from neutherm.training.dataset import (
    ParametricDataset,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from neutherm.training.losses import WeightedMSELoss, relative_l2_error

__all__ = [
    "ParametricDataset",
    "generate_dataset",
    "load_dataset",
    "save_dataset",
    "WeightedMSELoss",
    "relative_l2_error",
]
"""
Error metrics for evaluating and comparing model predictions.

Provides L2, Linf, and pointwise relative error functions
used by the comparison script to benchmark surrogate vs PINN vs solver.
"""

import numpy as np


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 error: ||pred - ref||_2 / ||ref||_2."""
    return float(np.linalg.norm(prediction - reference) / (np.linalg.norm(reference) + 1e-30))


def relative_linf(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative L-infinity error: max|pred - ref| / max|ref|."""
    return float(np.max(np.abs(prediction - reference)) / (np.max(np.abs(reference)) + 1e-30))


def pointwise_relative_error(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pointwise relative error: |pred - ref| / |ref| at each point."""
    return np.abs(prediction - reference) / (np.abs(reference) + 1e-30)


def mean_absolute_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(prediction - reference)))

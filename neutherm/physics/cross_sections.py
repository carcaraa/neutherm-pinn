"""
Temperature-dependent macroscopic cross sections with Doppler feedback.

This module provides functions to compute neutron cross sections as a function
of temperature, using the square-root Doppler broadening model:

    Σ(T) = Σ(T_ref) * (1 + α * sqrt(T - T_ref))

This is a simplified but widely used parametrization that captures the dominant
effect of thermal motion on resonance absorption (Doppler broadening of the
238U resonances in the epithermal range).

Both NumPy and PyTorch implementations are provided:
- NumPy versions are used in the reference finite-difference solver.
- PyTorch versions are used in the PINN, enabling automatic differentiation
  through the cross-section model.

References
----------
.. [1] Duderstadt & Hamilton, "Nuclear Reactor Analysis" (1976), §4.3.
.. [2] Stacey, "Nuclear Reactor Physics" (2007), §6.2.

Notes
-----
The temperature coefficients α are typically negative for absorption and
fission cross sections in thermal reactors (negative Doppler coefficient),
which is the primary safety feedback mechanism in LWRs.
"""

from dataclasses import dataclass

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neutherm.physics.parameters import NeutronicsParams

if TYPE_CHECKING:
    import torch


@dataclass
class CrossSectionSet:
    """A complete set of two-group cross sections at a given temperature.

    This is a plain container returned by the NumPy evaluation functions.
    All values in [cm^-1] except diffusion coefficients in [cm].
    """

    D1: np.ndarray  # Fast diffusion coefficient
    D2: np.ndarray  # Thermal diffusion coefficient
    sigma_r1: np.ndarray  # Fast removal
    sigma_a2: np.ndarray  # Thermal absorption
    nu_sigma_f1: np.ndarray  # Fast fission production
    nu_sigma_f2: np.ndarray  # Thermal fission production
    sigma_s12: np.ndarray  # Down-scattering 1→2


def _doppler_factor_np(T: np.ndarray, T_ref: float, alpha: float) -> np.ndarray:
    """Compute the Doppler correction factor (NumPy).

    Parameters
    ----------
    T : np.ndarray
        Temperature field [K].
    T_ref : float
        Reference temperature [K].
    alpha : float
        Temperature coefficient.

    Returns
    -------
    np.ndarray
        Multiplicative correction factor: 1 + α * sqrt(T - T_ref).
    """
    dT = np.maximum(T - T_ref, 0.0)  # Guard against T < T_ref
    return 1.0 + alpha * np.sqrt(dT)


def _doppler_factor_torch(T: torch.Tensor, T_ref: float, alpha: float) -> torch.Tensor:
    """Compute the Doppler correction factor (PyTorch, differentiable).

    Parameters
    ----------
    T : torch.Tensor
        Temperature field [K]. Must have requires_grad=True for PINN usage.
    T_ref : float
        Reference temperature [K].
    alpha : float
        Temperature coefficient.

    Returns
    -------
    torch.Tensor
        Multiplicative correction factor: 1 + α * sqrt(T - T_ref).
    """
    dT = torch.clamp(T - T_ref, min=0.0)
    return 1.0 + alpha * torch.sqrt(dT + 1e-12)  # Small epsilon for grad stability


def evaluate_cross_sections_np(
    T: np.ndarray,
    params: NeutronicsParams,
) -> CrossSectionSet:
    """Evaluate all two-group cross sections at given temperatures (NumPy).

    The diffusion coefficients D1, D2 are assumed temperature-independent
    (a reasonable approximation since D is dominated by elastic scattering,
    which has weak temperature dependence in the resolved resonance range).

    The removal cross section Σ_{r,1} = Σ_{a,1} + Σ_{s,1→2} is also kept
    constant, as the fast-group absorption is small and the down-scattering
    is dominated by hydrogen moderation (weak T dependence).

    Parameters
    ----------
    T : np.ndarray
        Temperature field [K], shape (N,) or scalar.
    params : NeutronicsParams
        Reference cross-section values and temperature coefficients.

    Returns
    -------
    CrossSectionSet
        All cross sections evaluated at the given temperatures.
    """
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))

    # Temperature-independent quantities
    D1 = np.full_like(T, params.D1_ref)
    D2 = np.full_like(T, params.D2_ref)
    sigma_r1 = np.full_like(T, params.sigma_r1_ref)
    sigma_s12 = np.full_like(T, params.sigma_s12_ref)

    # Temperature-dependent quantities (Doppler model)
    sigma_a2 = params.sigma_a2_ref * _doppler_factor_np(T, params.T_ref, params.alpha_a2)
    nu_sigma_f1 = params.nu_sigma_f1_ref * _doppler_factor_np(T, params.T_ref, params.alpha_f1)
    nu_sigma_f2 = params.nu_sigma_f2_ref * _doppler_factor_np(T, params.T_ref, params.alpha_f2)

    return CrossSectionSet(
        D1=D1,
        D2=D2,
        sigma_r1=sigma_r1,
        sigma_a2=sigma_a2,
        nu_sigma_f1=nu_sigma_f1,
        nu_sigma_f2=nu_sigma_f2,
        sigma_s12=sigma_s12,
    )


def evaluate_cross_sections_torch(
    T: torch.Tensor,
    params: NeutronicsParams,
) -> dict[str, torch.Tensor]:
    """Evaluate all two-group cross sections at given temperatures (PyTorch).

    This version preserves the computational graph for automatic differentiation,
    which is essential for computing PDE residuals in the PINN.

    Parameters
    ----------
    T : torch.Tensor
        Temperature field [K].
    params : NeutronicsParams
        Reference cross-section values and temperature coefficients.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys: 'D1', 'D2', 'sigma_r1', 'sigma_a2',
        'nu_sigma_f1', 'nu_sigma_f2', 'sigma_s12'.
    """
    # Temperature-independent (detached constants)
    D1 = torch.full_like(T, params.D1_ref)
    D2 = torch.full_like(T, params.D2_ref)
    sigma_r1 = torch.full_like(T, params.sigma_r1_ref)
    sigma_s12 = torch.full_like(T, params.sigma_s12_ref)

    # Temperature-dependent (differentiable)
    sigma_a2 = params.sigma_a2_ref * _doppler_factor_torch(T, params.T_ref, params.alpha_a2)
    nu_sigma_f1 = params.nu_sigma_f1_ref * _doppler_factor_torch(T, params.T_ref, params.alpha_f1)
    nu_sigma_f2 = params.nu_sigma_f2_ref * _doppler_factor_torch(T, params.T_ref, params.alpha_f2)

    return {
        "D1": D1,
        "D2": D2,
        "sigma_r1": sigma_r1,
        "sigma_a2": sigma_a2,
        "nu_sigma_f1": nu_sigma_f1,
        "nu_sigma_f2": nu_sigma_f2,
        "sigma_s12": sigma_s12,
    }

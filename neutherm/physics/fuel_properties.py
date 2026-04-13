"""
Thermal properties of UO2 fuel and heat transfer models.

This module provides:
- Temperature-dependent thermal conductivity of UO2 fuel (MATPRO correlation)
- Volumetric heat generation from fission
- Boundary condition models (gap conductance, coolant convection)

Both NumPy and PyTorch implementations are provided.

References
----------
.. [3] Todreas & Kazimi, "Nuclear Systems I" (2012), Ch. 8.
.. [11] Fink, J. K. (2000). "Thermophysical properties of uranium dioxide."
        Journal of Nuclear Materials, 279(1), 1-18.
"""

import numpy as np
import torch

from neutherm.physics.parameters import ThermalParams


# =============================================================================
# UO2 Thermal Conductivity
# =============================================================================


def fuel_conductivity_np(T: np.ndarray, params: ThermalParams) -> np.ndarray:
    """Temperature-dependent thermal conductivity of UO2 fuel (NumPy).

    Uses the MATPRO/Fink correlation:
        k(T) = 1 / (A + B*T) + C * T^3

    The first term captures phonon-phonon scattering (dominant below ~1500 K),
    and the second term captures electronic/radiative transport at high T.

    Parameters
    ----------
    T : np.ndarray
        Temperature [K].
    params : ThermalParams
        Contains fuel_k_A, fuel_k_B, fuel_k_C coefficients.

    Returns
    -------
    np.ndarray
        Thermal conductivity [W/(m·K)].
    """
    return 1.0 / (params.fuel_k_A + params.fuel_k_B * T) + params.fuel_k_C * T**3


def fuel_conductivity_torch(T: torch.Tensor, params: ThermalParams) -> torch.Tensor:
    """Temperature-dependent thermal conductivity of UO2 fuel (PyTorch).

    Differentiable version for use in the PINN loss computation.

    Parameters
    ----------
    T : torch.Tensor
        Temperature [K].
    params : ThermalParams
        Contains fuel_k_A, fuel_k_B, fuel_k_C coefficients.

    Returns
    -------
    torch.Tensor
        Thermal conductivity [W/(m·K)].
    """
    return 1.0 / (params.fuel_k_A + params.fuel_k_B * T) + params.fuel_k_C * T**3


# =============================================================================
# Volumetric Heat Generation
# =============================================================================


def heat_generation_np(
    phi1: np.ndarray,
    phi2: np.ndarray,
    sigma_f1: np.ndarray,
    sigma_f2: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Volumetric heat generation rate from fission (NumPy).

    q'''(r) = κ_f * (Σ_{f,1} * φ_1 + Σ_{f,2} * φ_2)

    Note: Σ_f here is the *fission* cross section, not the *production*
    cross section (νΣ_f). We obtain it as Σ_f = νΣ_f / ν, but since ν
    cancels in the coupled system when we normalize fluxes, we can use
    νΣ_f directly and absorb the ν into the normalization.

    In practice, we assume the fluxes are normalized such that
    q''' = κ * (Σ_{f,1} * φ_1 + Σ_{f,2} * φ_2) gives the correct
    power density.

    Parameters
    ----------
    phi1, phi2 : np.ndarray
        Neutron flux in group 1 (fast) and group 2 (thermal).
    sigma_f1, sigma_f2 : np.ndarray
        Fission cross sections (or νΣ_f / ν) [cm^-1].
    kappa : float
        Energy per fission [J/fission].

    Returns
    -------
    np.ndarray
        Volumetric heat generation rate [W/cm^3].
    """
    return kappa * (sigma_f1 * phi1 + sigma_f2 * phi2)


def heat_generation_torch(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    sigma_f1: torch.Tensor,
    sigma_f2: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Volumetric heat generation rate from fission (PyTorch).

    Differentiable version for PINN residual computation.

    Parameters
    ----------
    phi1, phi2 : torch.Tensor
        Neutron flux in group 1 (fast) and group 2 (thermal).
    sigma_f1, sigma_f2 : torch.Tensor
        Fission cross sections [cm^-1].
    kappa : float
        Energy per fission [J/fission].

    Returns
    -------
    torch.Tensor
        Volumetric heat generation rate [W/cm^3].
    """
    return kappa * (sigma_f1 * phi1 + sigma_f2 * phi2)


# =============================================================================
# Boundary Temperature Models
# =============================================================================


def fuel_surface_temperature(
    q_linear: float,
    params: ThermalParams,
) -> float:
    """Compute the fuel surface temperature from the linear heat rate.

    Applies the thermal resistance model:
        T_fuel_surface = T_coolant + q' / (2π R_c h_conv) + q' / (2π R_f h_gap)

    This is a simplified model assuming uniform heat flux.

    Parameters
    ----------
    q_linear : float
        Linear heat rate [W/m].
    params : ThermalParams
        Thermal-hydraulic parameters.

    Returns
    -------
    float
        Fuel surface temperature [K].
    """
    # This is computed in the solver using the actual radial profiles.
    # Here we provide a quick estimate for initialization.
    T_surf = params.T_coolant + q_linear / (2 * np.pi * params.h_conv * 0.00475)
    T_surf += q_linear / (2 * np.pi * params.h_gap * 0.004096)
    return T_surf

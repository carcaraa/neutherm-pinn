"""
Tests for the physics module.

Validates:
- YAML config loading and validation
- Cross-section temperature dependence (correct sign, magnitude)
- UO2 thermal conductivity (known values)
- NumPy / PyTorch consistency
"""

import numpy as np
import pytest
import torch

from neutherm.physics.parameters import NeutronicsParams, ProblemConfig, ThermalParams
from neutherm.physics.cross_sections import (
    evaluate_cross_sections_np,
    evaluate_cross_sections_torch,
)
from neutherm.physics.fuel_properties import fuel_conductivity_np, fuel_conductivity_torch


# ============================================================================
# Configuration
# ============================================================================


class TestConfig:
    def test_default_config(self):
        """Default config should be valid."""
        config = ProblemConfig()
        config.validate()

    def test_load_yaml(self, tmp_path):
        """Load config from YAML and verify a known field."""
        yaml_content = """
geometry:
  r_fuel: 0.005
  r_clad: 0.006
  n_radial: 50
physics:
  T_ref: 300.0
  D1_ref: 1.3
  D2_ref: 0.22
  sigma_r1_ref: 0.027
  sigma_a2_ref: 0.08
  nu_sigma_f1_ref: 0.009
  nu_sigma_f2_ref: 0.14
  sigma_s12_ref: 0.018
  alpha_a2: -2.0e-4
  alpha_f2: -1.8e-4
  alpha_f1: -0.5e-4
  kappa_fission: 3.204e-11
  fuel_k_A: 0.0375
  fuel_k_B: 2.165e-4
  fuel_k_C: 4.715e-12
  T_coolant: 590.0
  h_gap: 6000.0
  h_conv: 35000.0
"""
        f = tmp_path / "test_config.yaml"
        f.write_text(yaml_content)
        config = ProblemConfig.from_yaml(f)

        assert config.geometry.r_fuel == pytest.approx(0.005)
        assert config.geometry.n_radial == 50
        assert config.thermal.T_coolant == pytest.approx(590.0)
        assert config.neutronics.D1_ref == pytest.approx(1.3)

    def test_invalid_geometry_raises(self):
        """r_fuel > r_clad should fail validation."""
        config = ProblemConfig()
        config.geometry.r_fuel = 0.01
        config.geometry.r_clad = 0.005
        with pytest.raises(ValueError, match="Invalid geometry"):
            config.validate()


# ============================================================================
# Cross Sections
# ============================================================================


class TestCrossSections:
    @pytest.fixture
    def params(self):
        return NeutronicsParams()

    def test_reference_temperature_returns_reference_values(self, params):
        """At T = T_ref, cross sections should equal reference values."""
        T = np.array([params.T_ref])
        xs = evaluate_cross_sections_np(T, params)

        assert xs.D1[0] == pytest.approx(params.D1_ref)
        assert xs.D2[0] == pytest.approx(params.D2_ref)
        assert xs.sigma_a2[0] == pytest.approx(params.sigma_a2_ref)
        assert xs.nu_sigma_f2[0] == pytest.approx(params.nu_sigma_f2_ref)

    def test_negative_doppler_coefficient(self, params):
        """With negative α, cross sections should decrease with temperature."""
        T_low = np.array([400.0])
        T_high = np.array([1200.0])

        xs_low = evaluate_cross_sections_np(T_low, params)
        xs_high = evaluate_cross_sections_np(T_high, params)

        # α_a2 < 0 → sigma_a2 decreases with T
        assert xs_high.sigma_a2[0] < xs_low.sigma_a2[0]

        # α_f2 < 0 → nu_sigma_f2 decreases with T
        assert xs_high.nu_sigma_f2[0] < xs_low.nu_sigma_f2[0]

    def test_numpy_torch_consistency(self, params):
        """NumPy and PyTorch implementations should give identical results."""
        T_np = np.array([600.0, 900.0, 1200.0])
        T_torch = torch.tensor([600.0, 900.0, 1200.0], dtype=torch.float64)

        xs_np = evaluate_cross_sections_np(T_np, params)
        xs_torch = evaluate_cross_sections_torch(T_torch, params)

        np.testing.assert_allclose(
            xs_torch["sigma_a2"].numpy(), xs_np.sigma_a2, rtol=1e-6
        )
        np.testing.assert_allclose(
            xs_torch["nu_sigma_f2"].numpy(), xs_np.nu_sigma_f2, rtol=1e-6
        )

    def test_torch_gradient_flows(self, params):
        """PyTorch cross sections should support backpropagation."""
        T = torch.tensor([800.0], dtype=torch.float64, requires_grad=True)
        xs = evaluate_cross_sections_torch(T, params)

        # Compute a scalar loss and backprop
        loss = xs["sigma_a2"].sum()
        loss.backward()

        assert T.grad is not None
        # With α_a2 < 0, d(sigma_a2)/dT < 0
        assert T.grad.item() < 0

    def test_vectorized_evaluation(self, params):
        """Should handle arrays of different temperatures."""
        T = np.linspace(300.0, 1500.0, 50)
        xs = evaluate_cross_sections_np(T, params)

        assert xs.sigma_a2.shape == (50,)
        assert xs.D1.shape == (50,)
        # All values should be positive
        assert np.all(xs.sigma_a2 > 0)
        assert np.all(xs.nu_sigma_f2 > 0)


# ============================================================================
# Fuel Properties
# ============================================================================


class TestFuelConductivity:
    @pytest.fixture
    def params(self):
        return ThermalParams()

    def test_known_uo2_conductivity(self, params):
        """UO2 conductivity at ~700 K should be roughly 3-5 W/(m·K)."""
        T = np.array([700.0])
        k = fuel_conductivity_np(T, params)
        assert 2.0 < k[0] < 6.0, f"UO2 conductivity at 700K = {k[0]}, expected 3-5 W/(m·K)"

    def test_conductivity_decreases_with_temperature(self, params):
        """UO2 conductivity should generally decrease with T (phonon regime)."""
        T_low = np.array([500.0])
        T_high = np.array([1500.0])

        k_low = fuel_conductivity_np(T_low, params)
        k_high = fuel_conductivity_np(T_high, params)

        # In the phonon regime (below ~1500K), k decreases with T
        assert k_high[0] < k_low[0]

    def test_numpy_torch_consistency(self, params):
        """NumPy and PyTorch conductivity should match."""
        T_np = np.array([600.0, 1000.0, 1500.0])
        T_torch = torch.tensor([600.0, 1000.0, 1500.0], dtype=torch.float64)

        k_np = fuel_conductivity_np(T_np, params)
        k_torch = fuel_conductivity_torch(T_torch, params)

        np.testing.assert_allclose(k_torch.numpy(), k_np, rtol=1e-10)

    def test_torch_gradient(self, params):
        """Conductivity should be differentiable."""
        T = torch.tensor([800.0], dtype=torch.float64, requires_grad=True)
        k = fuel_conductivity_torch(T, params)
        k.backward()

        assert T.grad is not None
        # dk/dT should be negative in phonon regime
        assert T.grad.item() < 0

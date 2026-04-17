"""
Training pipeline for the Physics-Informed Neural Network (PINN).

The PINN loss has three components:

    L_total = λ_pde * L_pde + λ_bc * L_bc + λ_data * L_data

Where:
- L_pde:  PDE residuals evaluated at interior collocation points
          (how well does the network satisfy the governing equations?)
- L_bc:   Boundary condition violations at r=0 and r=R
          (does the solution respect symmetry and surface conditions?)
- L_data: Optional data loss from a few solver solutions
          (helps guide the PINN when pure physics training is slow)

The PDE residuals are computed using PyTorch autograd:
    dφ/dr  = torch.autograd.grad(φ, r)
    d²φ/dr² = torch.autograd.grad(dφ/dr, r)

This gives EXACT derivatives of the network output w.r.t. the input,
not finite-difference approximations. This is the core innovation of PINNs.

References
----------
.. [4] Raissi et al., "Physics-informed neural networks" (2019). JCP.
.. [6] Elhareef & Wu, "PINN for nuclear reactor calculations" (2023). ANE.
.. [7] Maddu et al., "Inverse Dirichlet weighting for PINNs" (2022).
.. [10] Wang et al., "When and why PINNs fail to train" (2022). JCP.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neutherm.physics.parameters import ProblemConfig, NeutronicsParams, ThermalParams
from neutherm.physics.cross_sections import evaluate_cross_sections_torch
from neutherm.physics.fuel_properties import fuel_conductivity_torch
from neutherm.models.pinn import PINNModel


# =============================================================================
# Derivative utilities using autograd
# =============================================================================


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of outputs w.r.t. inputs using autograd.

    This is the core operation of PINNs: it gives us exact derivatives
    of the neural network output with respect to its input (r).

    Parameters
    ----------
    outputs : torch.Tensor
        Network output (e.g., φ₁), shape (N, 1).
    inputs : torch.Tensor
        Network input (r), shape (N, 1). Must have requires_grad=True.

    Returns
    -------
    torch.Tensor
        d(outputs)/d(inputs), shape (N, 1).
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,  # Needed for higher-order derivatives
        retain_graph=True,  # Keep the graph for subsequent grad calls
    )[0]


# =============================================================================
# PDE Residual computation
# =============================================================================


def compute_pde_residuals(
    model: PINNModel,
    r: torch.Tensor,
    k_eff: torch.Tensor,
    neutronics: NeutronicsParams,
    thermal: ThermalParams,
    T_base: float = 900.0,
) -> dict[str, torch.Tensor]:
    """Compute the PDE residuals at collocation points.

    Evaluates how well the network's output satisfies the three
    governing equations:

    1. Fast-group diffusion:
       -1/r d/dr(r D₁ dφ₁/dr) + Σ_r1 φ₁ - (1/k) (νΣ_f1 φ₁ + νΣ_f2 φ₂) = 0

    2. Thermal-group diffusion:
       -1/r d/dr(r D₂ dφ₂/dr) + Σ_a2 φ₂ - Σ_s12 φ₁ = 0

    3. Heat conduction:
       -1/r d/dr(r k(T) dT/dr) - q'''(r) = 0

    All derivatives are computed via autograd — no discretization.

    Parameters
    ----------
    model : PINNModel
        The PINN network.
    r : torch.Tensor
        Collocation points [cm], shape (N, 1). Must have requires_grad=True.
    k_eff : torch.Tensor
        Effective multiplication factor (learnable parameter), scalar.
    neutronics : NeutronicsParams
        Cross-section parameters.
    thermal : ThermalParams
        Thermal parameters.
    T_base : float
        Base temperature [K] added to network output for T.
        The network predicts T_deviation, actual T = T_base + T_deviation.

    Returns
    -------
    dict[str, torch.Tensor]
        'R_fast': fast-group residual, shape (N, 1)
        'R_thermal': thermal-group residual, shape (N, 1)
        'R_heat': heat conduction residual, shape (N, 1)
    """
    # Forward pass through the network
    output = model(r)
    phi1 = output["phi1"]  # (N, 1)
    phi2 = output["phi2"]  # (N, 1)
    T_raw = output["T"]    # (N, 1) — network output (deviation from T_base)

    # Actual temperature = base + deviation
    T = T_base + T_raw

    # =========================================================================
    # Compute derivatives via autograd
    # =========================================================================
    # First derivatives
    dphi1_dr = grad(phi1, r)   # dφ₁/dr
    dphi2_dr = grad(phi2, r)   # dφ₂/dr
    dT_dr = grad(T_raw, r)     # dT/dr (use T_raw since T_base is constant)

    # Second derivatives
    d2phi1_dr2 = grad(dphi1_dr, r)  # d²φ₁/dr²
    d2phi2_dr2 = grad(dphi2_dr, r)  # d²φ₂/dr²
    d2T_dr2 = grad(dT_dr, r)        # d²T/dr²

    # =========================================================================
    # Evaluate temperature-dependent cross sections
    # =========================================================================
    # These are differentiable (PyTorch ops) so gradients flow through
    T_flat = T.squeeze()  # (N,) for the cross-section functions
    xs = evaluate_cross_sections_torch(T_flat, neutronics)

    D1 = xs["D1"].unsqueeze(1)          # (N, 1)
    D2 = xs["D2"].unsqueeze(1)
    sigma_r1 = xs["sigma_r1"].unsqueeze(1)
    sigma_a2 = xs["sigma_a2"].unsqueeze(1)
    nu_sf1 = xs["nu_sigma_f1"].unsqueeze(1)
    nu_sf2 = xs["nu_sigma_f2"].unsqueeze(1)
    sigma_s12 = xs["sigma_s12"].unsqueeze(1)

    # Thermal conductivity (differentiable)
    k_th = fuel_conductivity_torch(T_flat, thermal).unsqueeze(1)  # (N, 1)

    # dk/dr via chain rule: dk/dr = (dk/dT) * (dT/dr)
    # But we compute the full divergence term directly:
    # -1/r d/dr(r k dT/dr) = -k d²T/dr² - (k/r + dk/dr) dT/dr
    # Instead, use the product rule on r*k*dT/dr:
    # d/dr(r k dT/dr) = k dT/dr + r (dk/dT dT/dr dT/dr + k d²T/dr²)
    #                 = k dT/dr + r dk/dT (dT/dr)² + r k d²T/dr²
    # Simpler: compute dk/dT via autograd
    dk_dT = grad(k_th, T_raw)  # dk/dT through the chain

    # =========================================================================
    # Assemble PDE residuals
    # =========================================================================
    # Use r_safe to avoid division by zero at r=0
    # (the r=0 point is handled separately as a boundary condition)
    r_safe = torch.clamp(r, min=1e-6)

    # --- Fast-group diffusion residual ---
    # -1/r d/dr(r D₁ dφ₁/dr) + Σ_r1 φ₁ = (1/k) (νΣ_f1 φ₁ + νΣ_f2 φ₂)
    # Expanding: -D₁(d²φ₁/dr² + (1/r)dφ₁/dr) + Σ_r1 φ₁ - (1/k)(νΣ_f1 φ₁ + νΣ_f2 φ₂)
    # (D₁ is constant w.r.t. r in each region, so dD₁/dr = 0 within fuel)
    diffusion1 = -D1 * (d2phi1_dr2 + dphi1_dr / r_safe)
    fission_source = (1.0 / k_eff) * (nu_sf1 * phi1 + nu_sf2 * phi2)
    R_fast = diffusion1 + sigma_r1 * phi1 - fission_source

    # --- Thermal-group diffusion residual ---
    # -D₂(d²φ₂/dr² + (1/r)dφ₂/dr) + Σ_a2 φ₂ - Σ_s12 φ₁ = 0
    diffusion2 = -D2 * (d2phi2_dr2 + dphi2_dr / r_safe)
    R_thermal = diffusion2 + sigma_a2 * phi2 - sigma_s12 * phi1

    # --- Heat conduction residual ---
    # -1/r d/dr(r k(T) dT/dr) = q'''
    # Expanding: -(k d²T/dr² + (k/r + dk/dT dT/dr) dT/dr) = q'''
    heat_diffusion = -(k_th * d2T_dr2 + (k_th / r_safe + dk_dT * dT_dr) * dT_dr)

    # Heat generation: q''' = κ (Σ_f1 φ₁ + Σ_f2 φ₂)
    # Note: we use νΣ_f here and absorb ν into normalization (same as solver)
    kappa = thermal.kappa_fission
    q_triple_prime = kappa * (nu_sf1 * phi1 + nu_sf2 * phi2)

    # Convert q''' from W/cm³ to W/m³ (factor 10⁶) then back — actually
    # since both sides are in the same units within the fuel, keep CGS
    # But k_th is in W/(m·K) and r is in cm. We need consistent units.
    # Convert k_th to W/(cm·K): divide by 100
    # Then: -k[W/(cm·K)] * d²T/dr²[K/cm²] = q'''[W/cm³]  ✓
    k_th_cgs = k_th / 100.0  # W/(m·K) → W/(cm·K)
    dk_dT_cgs = dk_dT / 100.0

    heat_diffusion_cgs = -(
        k_th_cgs * d2T_dr2
        + (k_th_cgs / r_safe + dk_dT_cgs * dT_dr) * dT_dr
    )
    R_heat = heat_diffusion_cgs - q_triple_prime

    return {
        "R_fast": R_fast,
        "R_thermal": R_thermal,
        "R_heat": R_heat,
        "phi1": phi1,
        "phi2": phi2,
        "T": T,
    }


# =============================================================================
# Boundary condition losses
# =============================================================================


def compute_bc_loss(
    model: PINNModel,
    r_fuel_cm: float,
    T_surface: float,
    T_base: float,
) -> torch.Tensor:
    """Compute the boundary condition loss.

    Two boundary conditions:
    1. r = 0 (symmetry): dφ₁/dr = 0, dφ₂/dr = 0, dT/dr = 0
    2. r = R_fuel (fuel surface): T = T_surface

    We don't enforce φ at the boundary explicitly because the
    diffusion equation + eigenvalue will determine the flux shape.

    Parameters
    ----------
    model : PINNModel
        The PINN network.
    r_fuel_cm : float
        Fuel radius [cm].
    T_surface : float
        Prescribed fuel surface temperature [K].
    T_base : float
        Base temperature used in the PINN output.

    Returns
    -------
    torch.Tensor
        Scalar boundary condition loss.
    """
    device = next(model.parameters()).device

    # --- r = 0: symmetry (zero gradient) ---
    r0 = torch.zeros(1, 1, device=device, requires_grad=True)
    out0 = model(r0)

    dphi1_dr0 = grad(out0["phi1"], r0)
    dphi2_dr0 = grad(out0["phi2"], r0)
    dT_dr0 = grad(out0["T"], r0)

    bc_symmetry = (
        torch.mean(dphi1_dr0 ** 2)
        + torch.mean(dphi2_dr0 ** 2)
        + torch.mean(dT_dr0 ** 2)
    )

    # --- r = R_fuel: prescribed temperature ---
    r_surf = torch.full((1, 1), r_fuel_cm, device=device, requires_grad=True)
    out_surf = model(r_surf)
    T_pred_surface = T_base + out_surf["T"]
    bc_temperature = torch.mean((T_pred_surface - T_surface) ** 2)

    return bc_symmetry + bc_temperature


# =============================================================================
# Training history
# =============================================================================


@dataclass
class PINNHistory:
    """Records PINN training metrics across epochs."""

    total_loss: list[float] = field(default_factory=list)
    pde_loss: list[float] = field(default_factory=list)
    bc_loss: list[float] = field(default_factory=list)
    data_loss: list[float] = field(default_factory=list)
    k_eff_history: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_loss: float = float("inf")


# =============================================================================
# Main training function
# =============================================================================


def train_pinn(
    config: ProblemConfig,
    reference_solution=None,
    device: str = "auto",
    verbose: bool = True,
) -> tuple[PINNModel, PINNHistory, torch.Tensor]:
    """Train the PINN on the coupled neutronics-thermal problem.

    The PINN solves for a SINGLE set of physical parameters (not a
    parametric sweep like the surrogate). It finds φ₁(r), φ₂(r), T(r)
    that satisfy the PDEs, BCs, and optionally match reference data.

    k_eff is treated as a LEARNABLE PARAMETER (not a network output).
    This is standard for eigenvalue problems with PINNs: the network
    learns the eigenfunction (flux shape) while k_eff is optimized
    simultaneously as a separate scalar.

    Parameters
    ----------
    config : ProblemConfig
        Problem configuration.
    reference_solution : CoupledSolution, optional
        If provided, adds a data loss term comparing PINN output to
        the reference solver solution. This is the "hybrid" approach.
    device : str
        "auto", "cuda", or "cpu".
    verbose : bool
        Print training progress.

    Returns
    -------
    model : PINNModel
        Trained PINN model.
    history : PINNHistory
        Training metrics.
    k_eff : torch.Tensor
        Learned effective multiplication factor.
    """
    pinn_cfg = config.pinn
    geom = config.geometry
    neutronics = config.neutronics
    thermal = config.thermal

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    R_fuel_cm = geom.r_fuel * 100  # [m] → [cm]

    # =========================================================================
    # Build the PINN model
    # =========================================================================
    model = PINNModel(
        hidden_layers=pinn_cfg.hidden_layers,
        activation=pinn_cfg.activation,
        r_max=R_fuel_cm,
    ).to(device)

    # k_eff as a learnable parameter, initialized to 1.0
    # nn.Parameter tells PyTorch to include it in the optimizer
    k_eff = nn.Parameter(torch.tensor(1.0, device=device))

    # T_base: a reasonable starting temperature for the fuel
    # The network predicts deviations from this value
    T_base = float(thermal.T_coolant + 300.0)  # Rough estimate of average T

    # Estimate fuel surface temperature for BC
    # (simplified: T_coolant + some delta from gap/convection resistance)
    T_surface = thermal.T_coolant + 200.0  # Rough, will be refined

    if verbose:
        print("=" * 70)
        print("PINN Training")
        print("=" * 70)
        print(f"  Device: {device}")
        print(f"  Parameters: {model.count_parameters():,} (network) + 1 (k_eff)")
        print(f"  Hidden layers: {pinn_cfg.hidden_layers}")
        print(f"  Collocation points: {pinn_cfg.n_collocation}")
        print(f"  Epochs: {pinn_cfg.epochs}")
        print(f"  T_base: {T_base:.0f} K")
        print(f"  T_surface (BC): {T_surface:.0f} K")
        print("-" * 70)

    # =========================================================================
    # Optimizer: Adam for both network weights AND k_eff
    # =========================================================================
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": pinn_cfg.learning_rate},
            {"params": [k_eff], "lr": pinn_cfg.learning_rate * 10},
            # k_eff gets a higher LR because it's a single scalar that
            # needs to move faster than the thousands of network weights
        ],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pinn_cfg.epochs, eta_min=1e-6
    )

    # Loss weights
    lambda_pde = pinn_cfg.lambda_pde
    lambda_bc = pinn_cfg.lambda_bc
    lambda_data = pinn_cfg.lambda_data

    # =========================================================================
    # Prepare reference data (if provided)
    # =========================================================================
    has_data = reference_solution is not None
    if has_data:
        # Use the fuel-region solution as reference
        n_data = min(50, len(reference_solution.r_fuel))
        idx = np.linspace(0, len(reference_solution.r_fuel) - 1, n_data, dtype=int)
        r_data = torch.tensor(
            reference_solution.r_fuel[idx], dtype=torch.float32
        ).unsqueeze(1).to(device)
        phi1_data = torch.tensor(
            reference_solution.phi1[:len(reference_solution.r_fuel)][idx],
            dtype=torch.float32,
        ).unsqueeze(1).to(device)
        phi2_data = torch.tensor(
            reference_solution.phi2[:len(reference_solution.r_fuel)][idx],
            dtype=torch.float32,
        ).unsqueeze(1).to(device)
        T_data = torch.tensor(
            reference_solution.temperature[idx], dtype=torch.float32
        ).unsqueeze(1).to(device)

        # Normalization factors for data loss (so all fields contribute equally)
        phi1_scale = phi1_data.abs().max() + 1e-10
        phi2_scale = phi2_data.abs().max() + 1e-10
        T_scale = T_data.abs().max() + 1e-10

        if verbose:
            print(f"  Reference data: {n_data} points")
            print(f"  k_eff (reference): {reference_solution.k_eff:.6f}")

    # =========================================================================
    # Training loop
    # =========================================================================
    history = PINNHistory()
    best_state = None

    for epoch in range(1, pinn_cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # === Sample collocation points ===
        # Random points in the interior of the fuel domain (0, R_fuel)
        # We avoid r=0 exactly (singularity in 1/r) and handle it via BC
        r_colloc = (
            torch.rand(pinn_cfg.n_collocation, 1, device=device) * R_fuel_cm * 0.98
            + R_fuel_cm * 0.01
        )
        r_colloc.requires_grad_(True)

        # === PDE residuals ===
        residuals = compute_pde_residuals(
            model, r_colloc, k_eff, neutronics, thermal, T_base
        )

        # Normalize residuals by their characteristic scales
        # This prevents one equation from dominating the others
        loss_pde_fast = torch.mean(residuals["R_fast"] ** 2)
        loss_pde_thermal = torch.mean(residuals["R_thermal"] ** 2)
        loss_pde_heat = torch.mean(residuals["R_heat"] ** 2)

        # Normalize each residual by the magnitude of its dominant term
        # to make them comparable
        with torch.no_grad():
            scale_fast = torch.mean(residuals["phi1"] ** 2).clamp(min=1e-10)
            scale_thermal = torch.mean(residuals["phi2"] ** 2).clamp(min=1e-10)
            scale_heat = torch.mean(residuals["T"] ** 2).clamp(min=1e-10)

        loss_pde = (
            loss_pde_fast / scale_fast
            + loss_pde_thermal / scale_thermal
            + loss_pde_heat / scale_heat
        )

        # === Boundary conditions ===
        loss_bc = compute_bc_loss(model, R_fuel_cm, T_surface, T_base)

        # === Data loss (optional) ===
        if has_data:
            r_data_grad = r_data.clone().requires_grad_(True)
            out_data = model(r_data_grad)
            T_pred = T_base + out_data["T"]

            loss_data = (
                torch.mean(((out_data["phi1"] - phi1_data) / phi1_scale) ** 2)
                + torch.mean(((out_data["phi2"] - phi2_data) / phi2_scale) ** 2)
                + torch.mean(((T_pred - T_data) / T_scale) ** 2)
            )
        else:
            loss_data = torch.tensor(0.0, device=device)

        # === Total loss ===
        loss_total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_data * loss_data

        # === Backward + optimize ===
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # === Record history ===
        history.total_loss.append(loss_total.item())
        history.pde_loss.append(loss_pde.item())
        history.bc_loss.append(loss_bc.item())
        history.data_loss.append(loss_data.item())
        history.k_eff_history.append(k_eff.item())
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        # Save best model
        if loss_total.item() < history.best_loss:
            history.best_loss = loss_total.item()
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_keff = k_eff.item()

        # === Print progress ===
        if verbose and (epoch % max(1, pinn_cfg.epochs // 20) == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d}/{pinn_cfg.epochs}  "
                f"total={loss_total.item():.6f}  "
                f"pde={loss_pde.item():.4f}  "
                f"bc={loss_bc.item():.4f}  "
                f"k_eff={k_eff.item():.5f}  "
                f"lr={lr:.2e}"
            )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        k_eff_final = best_keff
    else:
        k_eff_final = k_eff.item()

    if verbose:
        print("-" * 70)
        print(f"  Best epoch: {history.best_epoch}")
        print(f"  Best loss: {history.best_loss:.6f}")
        print(f"  Learned k_eff: {k_eff_final:.6f}")
        if has_data:
            print(f"  Reference k_eff: {reference_solution.k_eff:.6f}")
            print(f"  k_eff error: {abs(k_eff_final - reference_solution.k_eff):.6f}")
        print("=" * 70)

    return model, history, torch.tensor(k_eff_final)


def save_pinn(
    model: PINNModel,
    history: PINNHistory,
    k_eff: torch.Tensor,
    path: str | Path,
):
    """Save the trained PINN model, history, and learned k_eff."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "k_eff": k_eff.item(),
            "history": {
                "total_loss": history.total_loss,
                "pde_loss": history.pde_loss,
                "bc_loss": history.bc_loss,
                "data_loss": history.data_loss,
                "k_eff_history": history.k_eff_history,
                "learning_rates": history.learning_rates,
                "best_epoch": history.best_epoch,
                "best_loss": history.best_loss,
            },
        },
        path,
    )
    print(f"PINN saved to {path}")


def plot_pinn_training(history: PINNHistory, save_path: str | Path | None = None):
    """Plot PINN training curves: total loss, PDE/BC components, and k_eff evolution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history.total_loss) + 1)

    # Loss components
    axes[0].semilogy(epochs, history.total_loss, label="Total", alpha=0.7)
    axes[0].semilogy(epochs, history.pde_loss, label="PDE", alpha=0.7)
    axes[0].semilogy(epochs, history.bc_loss, label="BC", alpha=0.7)
    if any(v > 0 for v in history.data_loss):
        axes[0].semilogy(epochs, history.data_loss, label="Data", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("PINN loss components")
    axes[0].legend()

    # k_eff evolution
    axes[1].plot(epochs, history.k_eff_history, color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("k_eff")
    axes[1].set_title("Learned k_eff")

    # Learning rate
    axes[2].plot(epochs, history.learning_rates, color="tab:orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning rate schedule")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"PINN training curves saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the PINN model.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--with-data", action="store_true",
        help="Use reference solver solution as additional training data.",
    )
    parser.add_argument(
        "--output", type=str, default="results/pinn_model.pt",
        help="Path to save the trained PINN.",
    )
    args = parser.parse_args()

    cfg = ProblemConfig.from_yaml(args.config)

    # Optionally generate reference data
    ref_solution = None
    if args.with_data:
        from neutherm.solvers.coupled_solver import solve_coupled
        print("Generating reference solution...")
        ref_solution = solve_coupled(cfg, power_level=200.0, verbose=False)
        print(f"Reference k_eff = {ref_solution.k_eff:.6f}")

    # Train the PINN
    model, history, k_eff = train_pinn(
        cfg, reference_solution=ref_solution, verbose=True
    )

    # Save
    save_pinn(model, history, k_eff, args.output)
    plot_pinn_training(history, save_path="results/pinn_training.png")

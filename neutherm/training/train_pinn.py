"""
Training pipeline for the PINN on the full pin cell domain.

The PINN loss has three components:
    L = λ_pde * L_pde + λ_bc * L_bc + λ_data * L_data

Key change from the fuel-only version: the PDE residuals now handle
TWO REGIONS with different physics:
- Fuel [0, R_fuel]: diffusion + fission + heat generation
- Moderator [R_fuel, R_cell]: diffusion + absorption (no fission, no heat)

Boundary conditions:
- r = 0: symmetry (dφ/dr = 0, dT/dr = 0)
- r = R_cell: reflective (dφ/dr = 0) — matches the solver's Wigner-Seitz BC
- r = R_fuel: continuity of flux and temperature (handled implicitly
  by the continuous neural network)

References
----------
.. [4] Raissi et al., "Physics-informed neural networks" (2019). JCP.
.. [6] Elhareef & Wu, "PINN for nuclear reactor calculations" (2023). ANE.
.. [7] Maddu et al., "Inverse Dirichlet weighting for PINNs" (2022).
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
# Autograd derivative utilities
# =============================================================================


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute d(outputs)/d(inputs) via autograd. Creates graph for higher-order derivs."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# =============================================================================
# PDE residuals for the full pin cell
# =============================================================================


def compute_fuel_residuals(
    model: PINNModel,
    r: torch.Tensor,
    k_eff: torch.Tensor,
    neutronics: NeutronicsParams,
    thermal: ThermalParams,
    T_base: float,
) -> dict[str, torch.Tensor]:
    """PDE residuals in the FUEL region: diffusion + fission + heat conduction.

    Three equations:
    1. Fast diffusion:   -D₁ (d²φ₁/dr² + 1/r dφ₁/dr) + Σ_r1 φ₁ = (1/k)(νΣ_f1 φ₁ + νΣ_f2 φ₂)
    2. Thermal diffusion: -D₂ (d²φ₂/dr² + 1/r dφ₂/dr) + Σ_a2 φ₂ = Σ_s12 φ₁
    3. Heat conduction:  -(k d²T/dr² + (k/r + dk/dT dT/dr) dT/dr) = κ (νΣ_f1 φ₁ + νΣ_f2 φ₂)
    """
    out = model(r)
    phi1, phi2, T_raw = out["phi1"], out["phi2"], out["T"]
    T = T_base + T_raw

    # Derivatives via autograd
    dphi1 = grad(phi1, r)
    dphi2 = grad(phi2, r)
    dT = grad(T_raw, r)
    d2phi1 = grad(dphi1, r)
    d2phi2 = grad(dphi2, r)
    d2T = grad(dT, r)

    # Cross sections (temperature-dependent, differentiable)
    T_flat = T.squeeze()
    xs = evaluate_cross_sections_torch(T_flat, neutronics)
    D1 = xs["D1"].unsqueeze(1)
    D2 = xs["D2"].unsqueeze(1)
    sigma_r1 = xs["sigma_r1"].unsqueeze(1)
    sigma_a2 = xs["sigma_a2"].unsqueeze(1)
    nu_sf1 = xs["nu_sigma_f1"].unsqueeze(1)
    nu_sf2 = xs["nu_sigma_f2"].unsqueeze(1)
    sigma_s12 = xs["sigma_s12"].unsqueeze(1)

    # Thermal conductivity and its derivative
    k_th = fuel_conductivity_torch(T_flat, thermal).unsqueeze(1)
    dk_dT = grad(k_th, T_raw)

    # Convert k from W/(m·K) to W/(cm·K) for unit consistency with r in [cm]
    k_cgs = k_th / 100.0
    dk_cgs = dk_dT / 100.0

    r_safe = torch.clamp(r, min=1e-6)

    # Fast group residual
    diff1 = -D1 * (d2phi1 + dphi1 / r_safe)
    fission = (1.0 / k_eff) * (nu_sf1 * phi1 + nu_sf2 * phi2)
    R_fast = diff1 + sigma_r1 * phi1 - fission

    # Thermal group residual
    diff2 = -D2 * (d2phi2 + dphi2 / r_safe)
    R_thermal = diff2 + sigma_a2 * phi2 - sigma_s12 * phi1

    # Heat conduction residual
    heat_diff = -(k_cgs * d2T + (k_cgs / r_safe + dk_cgs * dT) * dT)
    q_source = thermal.kappa_fission * (nu_sf1 * phi1 + nu_sf2 * phi2)
    R_heat = heat_diff - q_source

    return {
        "R_fast": R_fast, "R_thermal": R_thermal, "R_heat": R_heat,
        "phi1": phi1, "phi2": phi2, "T": T,
    }


def compute_moderator_residuals(
    model: PINNModel,
    r: torch.Tensor,
    neutronics: NeutronicsParams,
    T_base: float,
) -> dict[str, torch.Tensor]:
    """PDE residuals in the MODERATOR region: diffusion only, no fission, no heat.

    Two equations (no heat equation — moderator temperature is fixed):
    1. Fast diffusion:    -D₁_mod (d²φ₁/dr² + 1/r dφ₁/dr) + Σ_r1_mod φ₁ = 0
    2. Thermal diffusion: -D₂_mod (d²φ₂/dr² + 1/r dφ₂/dr) + Σ_a2_mod φ₂ = Σ_s12_mod φ₁

    Cross sections are CONSTANT in the moderator (no temperature dependence).
    """
    out = model(r)
    phi1, phi2 = out["phi1"], out["phi2"]

    dphi1 = grad(phi1, r)
    dphi2 = grad(phi2, r)
    d2phi1 = grad(dphi1, r)
    d2phi2 = grad(dphi2, r)

    # Moderator cross sections (constant)
    D1 = neutronics.D1_mod
    D2 = neutronics.D2_mod
    sigma_r1 = neutronics.sigma_r1_mod
    sigma_a2 = neutronics.sigma_a2_mod
    sigma_s12 = neutronics.sigma_s12_mod

    r_safe = torch.clamp(r, min=1e-6)

    # Fast group: no fission source in moderator
    R_fast = -D1 * (d2phi1 + dphi1 / r_safe) + sigma_r1 * phi1

    # Thermal group: scattering source from fast group
    R_thermal = -D2 * (d2phi2 + dphi2 / r_safe) + sigma_a2 * phi2 - sigma_s12 * phi1

    return {"R_fast": R_fast, "R_thermal": R_thermal, "phi1": phi1, "phi2": phi2}


# =============================================================================
# Boundary condition losses
# =============================================================================


def compute_bc_loss(
    model: PINNModel,
    r_cell_cm: float,
    T_surface: float,
    T_base: float,
) -> torch.Tensor:
    """Boundary conditions for the full pin cell.

    1. r = 0: symmetry → dφ₁/dr = 0, dφ₂/dr = 0, dT/dr = 0
    2. r = R_cell: reflective → dφ₁/dr = 0, dφ₂/dr = 0
    3. r = R_fuel: T = T_surface (fuel surface temperature from gap model)
    """
    device = next(model.parameters()).device

    # --- r = 0: symmetry ---
    r0 = torch.zeros(1, 1, device=device, requires_grad=True)
    out0 = model(r0)
    bc_sym = (
        torch.mean(grad(out0["phi1"], r0) ** 2)
        + torch.mean(grad(out0["phi2"], r0) ** 2)
        + torch.mean(grad(out0["T"], r0) ** 2)
    )

    # --- r = R_cell: reflective (zero net current) ---
    r_cell = torch.full((1, 1), r_cell_cm, device=device, requires_grad=True)
    out_cell = model(r_cell)
    bc_refl = (
        torch.mean(grad(out_cell["phi1"], r_cell) ** 2)
        + torch.mean(grad(out_cell["phi2"], r_cell) ** 2)
    )

    # --- r = R_fuel: prescribed temperature ---
    r_fuel = torch.full(
        (1, 1), model.r_fuel, device=device, requires_grad=True
    )
    out_fuel = model(r_fuel)
    T_pred = T_base + out_fuel["T"]
    bc_temp = torch.mean((T_pred - T_surface) ** 2)

    return bc_sym + bc_refl + bc_temp


# =============================================================================
# Training history
# =============================================================================


@dataclass
class PINNHistory:
    """Records PINN training metrics."""
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
    """Train the PINN on the full pin cell domain.

    Collocation points are sampled in BOTH fuel and moderator regions.
    Fuel points get fuel-physics residuals (diffusion + fission + heat).
    Moderator points get moderator-physics residuals (diffusion only).
    k_eff is a learnable parameter optimized jointly with the network.
    """
    pinn_cfg = config.pinn
    geom = config.geometry
    neutronics = config.neutronics
    thermal = config.thermal

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    R_fuel_cm = geom.r_fuel * 100
    R_cell_cm = geom.r_cell * 100

    # Fraction of collocation points in each region (proportional to area)
    fuel_area = R_fuel_cm ** 2
    cell_area = R_cell_cm ** 2
    fuel_frac = fuel_area / cell_area
    n_fuel_pts = int(pinn_cfg.n_collocation * fuel_frac)
    n_mod_pts = pinn_cfg.n_collocation - n_fuel_pts

    # Build model over full pin cell
    model = PINNModel(
        hidden_layers=pinn_cfg.hidden_layers,
        activation=pinn_cfg.activation,
        r_fuel=R_fuel_cm,
        r_cell=R_cell_cm,
    ).to(device)

    # k_eff: learnable scalar, initialized to 1.0
    k_eff = nn.Parameter(torch.tensor(1.0, device=device))

    T_base = float(thermal.T_coolant + 300.0)
    T_surface = thermal.T_coolant + 200.0

    if verbose:
        print("=" * 70)
        print("PINN Training (Full Pin Cell)")
        print("=" * 70)
        print(f"  Device: {device}")
        print(f"  Parameters: {model.count_parameters():,} + 1 (k_eff)")
        print(f"  Domain: [0, {R_cell_cm:.4f}] cm (fuel: {R_fuel_cm:.4f} cm)")
        print(f"  Collocation: {n_fuel_pts} fuel + {n_mod_pts} moderator")
        print(f"  Epochs: {pinn_cfg.epochs}")
        print("-" * 70)

    # Optimizer: separate LR for k_eff (higher) and network (lower)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": pinn_cfg.learning_rate},
        {"params": [k_eff], "lr": pinn_cfg.learning_rate * 10},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pinn_cfg.epochs, eta_min=1e-6
    )

    lambda_pde = pinn_cfg.lambda_pde
    lambda_bc = pinn_cfg.lambda_bc
    lambda_data = pinn_cfg.lambda_data

    # --- Reference data (optional) ---
    has_data = reference_solution is not None
    if has_data:
        ref = reference_solution
        n_full = len(ref.r_neutronics)
        n_data = min(80, n_full)
        idx = np.linspace(0, n_full - 1, n_data, dtype=int)

        r_data = torch.tensor(ref.r_neutronics[idx], dtype=torch.float32).unsqueeze(1).to(device)
        phi1_data = torch.tensor(ref.phi1[idx], dtype=torch.float32).unsqueeze(1).to(device)
        phi2_data = torch.tensor(ref.phi2[idx], dtype=torch.float32).unsqueeze(1).to(device)

        # Temperature data: only available in fuel region
        n_fuel_data = config.geometry.n_radial
        idx_fuel = np.linspace(0, n_fuel_data - 1, min(50, n_fuel_data), dtype=int)
        r_data_fuel = torch.tensor(ref.r_fuel[idx_fuel], dtype=torch.float32).unsqueeze(1).to(device)
        T_data = torch.tensor(ref.temperature[idx_fuel], dtype=torch.float32).unsqueeze(1).to(device)

        # Normalization scales
        phi1_scale = phi1_data.abs().max().clamp(min=1e-10)
        phi2_scale = phi2_data.abs().max().clamp(min=1e-10)
        T_scale = T_data.abs().max().clamp(min=1e-10)

        if verbose:
            print(f"  Reference data: {n_data} flux pts + {len(idx_fuel)} temp pts")
            print(f"  Reference k_eff: {ref.k_eff:.6f}")

    # --- Training loop ---
    history = PINNHistory()
    best_state = None
    best_keff = 1.0

    for epoch in range(1, pinn_cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # === Sample collocation points ===
        # Fuel region: (0, R_fuel), avoiding r=0 singularity
        r_fuel_pts = (
            torch.rand(n_fuel_pts, 1, device=device) * R_fuel_cm * 0.98
            + R_fuel_cm * 0.01
        )
        r_fuel_pts.requires_grad_(True)

        # Moderator region: (R_fuel, R_cell)
        r_mod_pts = (
            torch.rand(n_mod_pts, 1, device=device) * (R_cell_cm - R_fuel_cm) * 0.98
            + R_fuel_cm * 1.01
        )
        r_mod_pts.requires_grad_(True)

        # === Fuel PDE residuals ===
        fuel_res = compute_fuel_residuals(
            model, r_fuel_pts, k_eff, neutronics, thermal, T_base
        )

        loss_fuel_fast = torch.mean(fuel_res["R_fast"] ** 2)
        loss_fuel_thermal = torch.mean(fuel_res["R_thermal"] ** 2)
        loss_fuel_heat = torch.mean(fuel_res["R_heat"] ** 2)

        # Normalize by field magnitudes (detached to avoid affecting gradients)
        with torch.no_grad():
            s_fast = torch.mean(fuel_res["phi1"] ** 2).clamp(min=1e-10)
            s_therm = torch.mean(fuel_res["phi2"] ** 2).clamp(min=1e-10)
            s_heat = torch.mean(fuel_res["T"] ** 2).clamp(min=1e-10)

        loss_pde_fuel = loss_fuel_fast / s_fast + loss_fuel_thermal / s_therm + loss_fuel_heat / s_heat

        # === Moderator PDE residuals ===
        mod_res = compute_moderator_residuals(model, r_mod_pts, neutronics, T_base)

        loss_mod_fast = torch.mean(mod_res["R_fast"] ** 2)
        loss_mod_thermal = torch.mean(mod_res["R_thermal"] ** 2)

        with torch.no_grad():
            sm_fast = torch.mean(mod_res["phi1"] ** 2).clamp(min=1e-10)
            sm_therm = torch.mean(mod_res["phi2"] ** 2).clamp(min=1e-10)

        loss_pde_mod = loss_mod_fast / sm_fast + loss_mod_thermal / sm_therm

        loss_pde = loss_pde_fuel + loss_pde_mod

        # === Boundary conditions ===
        loss_bc = compute_bc_loss(model, R_cell_cm, T_surface, T_base)

        # === Data loss (optional) ===
        if has_data:
            # Flux data (full cell mesh)
            r_d = r_data.clone().requires_grad_(True)
            out_d = model(r_d)
            loss_data = (
                torch.mean(((out_d["phi1"] - phi1_data) / phi1_scale) ** 2)
                + torch.mean(((out_d["phi2"] - phi2_data) / phi2_scale) ** 2)
            )
            # Temperature data (fuel only)
            r_df = r_data_fuel.clone().requires_grad_(True)
            out_df = model(r_df)
            T_pred = T_base + out_df["T"]
            loss_data = loss_data + torch.mean(((T_pred - T_data) / T_scale) ** 2)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # === Total loss ===
        loss_total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_data * loss_data

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # === Record ===
        history.total_loss.append(loss_total.item())
        history.pde_loss.append(loss_pde.item())
        history.bc_loss.append(loss_bc.item())
        history.data_loss.append(loss_data.item())
        history.k_eff_history.append(k_eff.item())
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        if loss_total.item() < history.best_loss:
            history.best_loss = loss_total.item()
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_keff = k_eff.item()

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

    if verbose:
        print("-" * 70)
        print(f"  Best epoch: {history.best_epoch}")
        print(f"  Learned k_eff: {best_keff:.6f}")
        if has_data:
            print(f"  Reference k_eff: {ref.k_eff:.6f}")
            print(f"  k_eff error: {abs(best_keff - ref.k_eff):.6f} "
                  f"({abs(best_keff - ref.k_eff)/ref.k_eff*100:.2f}%)")
        print("=" * 70)

    return model, history, torch.tensor(best_keff)


def save_pinn(model: PINNModel, history: PINNHistory, k_eff: torch.Tensor, path: str | Path):
    """Save trained PINN model, history, and learned k_eff."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "k_eff": k_eff.item(),
        "r_fuel": model.r_fuel,
        "r_cell": model.r_cell,
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
    }, path)
    print(f"PINN saved to {path}")


def plot_pinn_training(history: PINNHistory, save_path: str | Path | None = None):
    """Plot PINN training curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history.total_loss) + 1)

    axes[0].semilogy(epochs, history.total_loss, label="Total", alpha=0.7)
    axes[0].semilogy(epochs, history.pde_loss, label="PDE", alpha=0.7)
    axes[0].semilogy(epochs, history.bc_loss, label="BC", alpha=0.7)
    if any(v > 0 for v in history.data_loss):
        axes[0].semilogy(epochs, history.data_loss, label="Data", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("PINN loss components")
    axes[0].legend()

    axes[1].plot(epochs, history.k_eff_history, color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("k_eff")
    axes[1].set_title("Learned k_eff")

    axes[2].plot(epochs, history.learning_rates, color="tab:orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning rate schedule")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the PINN (full pin cell).")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--with-data", action="store_true",
                        help="Use reference solver solution as training data.")
    parser.add_argument("--output", type=str, default="results/pinn_model.pt")
    args = parser.parse_args()

    cfg = ProblemConfig.from_yaml(args.config)

    ref_solution = None
    if args.with_data:
        from neutherm.solvers.coupled_solver import solve_coupled
        print("Generating reference solution...")
        ref_solution = solve_coupled(cfg, power_level=200.0, verbose=False)
        print(f"Reference k_eff = {ref_solution.k_eff:.6f}")

    model, history, k_eff = train_pinn(cfg, reference_solution=ref_solution, verbose=True)
    save_pinn(model, history, k_eff, args.output)
    plot_pinn_training(history, save_path="results/pinn_training.png")
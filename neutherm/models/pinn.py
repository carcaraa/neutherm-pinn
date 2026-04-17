"""
Physics-Informed Neural Network for the coupled neutronics-thermal problem.

Unlike the surrogate (which maps parameters → field vectors), the PINN
represents the solution as a CONTINUOUS FUNCTION of radius r:

    NN(r) → (φ₁(r), φ₂(r), T(r))

Since the outputs are differentiable functions of r (via PyTorch autograd),
we can compute dφ/dr, d²φ/dr² analytically and evaluate PDE residuals
at arbitrary collocation points. No labeled data required — only the
governing equations and boundary conditions.

Why tanh and not ReLU?
    ReLU has zero second derivative everywhere (except at 0 where it's
    undefined). Since our PDEs contain d²φ/dr², a ReLU network gives
    d²φ/dr² ≈ 0 everywhere, making the PDE residual meaningless.
    Tanh is C^∞ (infinitely differentiable), so autograd produces
    meaningful higher-order derivatives.

References
----------
.. [4] Raissi et al., "Physics-informed neural networks" (2019). JCP.
.. [10] Wang et al., "When and why PINNs fail to train" (2022). JCP.
"""

import torch
import torch.nn as nn


class PINNModel(nn.Module):
    """Neural network that approximates the coupled solution as NN(r).

    The network takes a single scalar input r (radial position, normalized
    to [0, 1]) and outputs three fields: fast flux φ₁, thermal flux φ₂,
    and temperature T.

    The architecture is a simple fully connected network. We intentionally
    avoid skip connections here (unlike the surrogate) because the PINN
    needs smooth, well-behaved second derivatives. Skip connections can
    introduce kinks in the derivative landscape that confuse the PDE
    residual computation.

    Parameters
    ----------
    hidden_layers : list[int]
        Width of each hidden layer (e.g., [64, 64, 64, 64]).
    activation : str
        Activation function: "tanh" (recommended), "silu", or "gelu".
    r_max : float
        Maximum radius [cm] for input normalization.
        The network sees r_norm = r / r_max ∈ [0, 1].
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
        r_max: float = 1.0,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        self.r_max = r_max

        # Select activation — must be smooth for second derivatives
        activation_fns = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,   # Also smooth (C^∞)
            "gelu": nn.GELU,   # Approximately smooth
        }
        act_cls = activation_fns[activation.lower()]

        # Build the network layer by layer
        layers = []
        in_features = 1  # Single input: r

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(act_cls())
            in_features = width

        # Output layer: 3 values (φ₁, φ₂, T) — no activation
        # (the outputs can be any real number)
        layers.append(nn.Linear(in_features, 3))

        self.network = nn.Sequential(*layers)

        # Apply Xavier initialization for better training with tanh
        # Xavier init sets weights ~ U(-sqrt(6/(fan_in+fan_out)), ...) which
        # keeps the variance of activations stable across layers with tanh.
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for all linear layers."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: r → (φ₁, φ₂, T).

        Parameters
        ----------
        r : torch.Tensor
            Radial positions [cm], shape (N, 1). Must have requires_grad=True
            for PDE residual computation.

        Returns
        -------
        dict[str, torch.Tensor]
            'phi1': shape (N, 1), 'phi2': shape (N, 1), 'T': shape (N, 1).
        """
        # Normalize r to [0, 1] for better training
        r_norm = r / self.r_max

        # Forward through the network
        out = self.network(r_norm)  # shape (N, 3)

        # Split into individual fields
        # We apply softplus to fluxes to ensure they're positive
        # (neutron flux is always ≥ 0 physically)
        phi1 = torch.nn.functional.softplus(out[:, 0:1])
        phi2 = torch.nn.functional.softplus(out[:, 1:2])

        # Temperature: shift by a base value to help the network
        # The network predicts a deviation from T_base, which we add back.
        # This makes the network's job easier: predict small deviations
        # instead of large absolute values (~1000 K).
        T = out[:, 2:3]

        return {"phi1": phi1, "phi2": phi2, "T": T}

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Numerical solvers for the coupled neutronics-thermal problem.

Submodules
----------
diffusion_solver : Two-group neutron diffusion (FD + power iteration).
thermal_solver : Radial heat conduction with T-dependent conductivity.
coupled_solver : Picard iteration coupling neutronics ↔ thermal.
"""

from neutherm.solvers.coupled_solver import CoupledSolution, solve_coupled
from neutherm.solvers.diffusion_solver import solve_diffusion
from neutherm.solvers.thermal_solver import solve_thermal

__all__ = [
    "solve_diffusion",
    "solve_thermal",
    "solve_coupled",
    "CoupledSolution",
]
"""
ECG Signal Imputation Operator - Clean Implementation

A differentiable signal imputation operator with dual-mode backward pass:
- Envelope mode: Direct gradient of value function
- Implicit mode: Adjoint gradient through optimizer

Core components:
- config: Configuration classes
- ops: Linear operators (D, D^T, A, A^T)
- weights: Safe weight parameterization
- solver: Conjugate gradient solver
- layer: Main autograd module
"""

from .config import CGConfig, LayerConfig
from .ops import apply_D, apply_DT, A_apply, AT_apply, make_known_full
from .weights import map_logits_to_w, WeightNet
from .solver import solve_forward, solve_adjoint
from .layer import DiagOpLayer

__version__ = "1.0.0"
__all__ = [
    "CGConfig", "LayerConfig",
    "apply_D", "apply_DT", "A_apply", "AT_apply", "make_known_full",
    "map_logits_to_w", "WeightNet",
    "solve_forward", "solve_adjoint",
    "DiagOpLayer"
]

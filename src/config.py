"""
Configuration classes for the ECG signal imputation operator.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CGConfig:
    """Configuration for Conjugate Gradient solver."""
    tol: float = 1e-6
    maxiter: int = 2000
    verbose: bool = False
    jitter: float = 0.0  # Numerical jitter for stability


@dataclass
class LayerConfig:
    """Configuration for the DiagOpLayer (implicit gradient only)."""
    mode: str = "implicit"  # implicit gradient only
    eps_w: float = 1e-4
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

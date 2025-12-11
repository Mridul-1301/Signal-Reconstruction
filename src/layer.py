"""
DiagOpLayer: Autograd module implementing implicit-gradient training only.

This layer exposes the value function φ(w) for monitoring, and supports
implicit gradients for outer objectives that depend on the optimizer solution.
"""

import torch
import torch.nn as nn
from typing import Dict
from .ops import A_apply, apply_D, make_known_full
from .weights import map_logits_to_w
from .solver import solve_forward, solve_adjoint
from .config import CGConfig, LayerConfig


class DiagOpLayer(nn.Module):
    """
    Computes φ(w) = min_v ||W(Av-b)||^2 and supports a general outer loss L(v*)
    by providing an implicit adjoint path when an external gradient g = dL/dv*
    is supplied.
    """
    
    def __init__(self, cg_config: CGConfig, layer_config: LayerConfig):
        super().__init__()
        self.cg_config = cg_config
        self.layer_config = layer_config
        
    def forward(self, u_logits: torch.Tensor, x_full: torch.Tensor, 
                unknown_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the layer.
        
        Args:
            u_logits: [B, m] logits -> weights
            x_full: [B, n] full signal (NaNs allowed; unknown_idx marks unknowns)
            unknown_idx: [n_miss] Long indices of unknown positions
                
        Returns:
            out: dict with keys:
                'phi': [B] scalar φ (value function)
                'v': [B, n_miss] optimizer
                'r': [B, m] residual Av-b
                'w': [B, m] weights
        """
        _, n = x_full.shape
        
        # Clean x_full (replace NaNs with zeros)
        x = torch.nan_to_num(x_full, nan=0.0)
        
        # Build S1 x_known and operators
        x_known = make_known_full(x, unknown_idx)  # [B, n]
        b = -apply_D(x_known)  # [B, m]
        
        # Map logits to weights
        w = map_logits_to_w(u_logits, eps=self.layer_config.eps_w,
                           clamp=(self.layer_config.clamp_min, self.layer_config.clamp_max))  # [B, m]
        
        # Forward solve: K(w) v = h(w)
        v, solve_info = solve_forward(w, b, unknown_idx, n, self.cg_config)
        
        # Compute residual and value function
        Av = A_apply(v, unknown_idx, n)  # [B, m]
        r = Av - b  # [B, m]
        phi = ((w * w) * (r * r)).sum(dim=1)  # [B]
        
        # Store for potential use in backward pass
        self._cache = {
            "r": r, "w": w, "v": v, "solve_info": solve_info, 
            "n": n, "unknown_idx": unknown_idx, "b": b
        }
        
        return {"phi": phi, "v": v, "r": r, "w": w}
    
    def compute_implicit_gradient(self, r: torch.Tensor, w: torch.Tensor, 
                                 g: torch.Tensor) -> torch.Tensor:
        """
        Compute implicit gradient: dL/dw = 2 w (r ⊙ A y) where K y = g.
        
        Args:
            r: [B, m] residual vector
            w: [B, m] weights
            g: [B, n_miss] external gradient
            
        Returns:
            [B, m] gradient with respect to weights
        """
        # Solve adjoint system: K y = g
        y, _ = solve_adjoint(w, g, self._cache["unknown_idx"], 
                            self._cache["n"], self.cg_config)
        
        # Compute A y
        Ay = A_apply(y, self._cache["unknown_idx"], self._cache["n"])  # [B, m]
        
        # Compute gradient: 2 w (r ⊙ A y)
        return 2.0 * w * (r * Ay)

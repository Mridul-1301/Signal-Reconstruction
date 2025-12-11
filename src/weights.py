"""
Safe weight parameterization to prevent weight collapse.

Maps logits u to positive weights w with mean scale ~1 to avoid trivial w→0 solutions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def map_logits_to_w(u_logits: torch.Tensor, eps: float = 1e-4, 
                   clamp: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Map logits to positive weights with mean scale ~1.
    
    Formula:
        w̃ = exp(u)
        w̄ = (1/m) * Σⱼ w̃ⱼ  
        w = ε + w̃ / (w̄ + 1e-12)
    
    This ensures w_i > 0 and fixes mean scale to prevent trivial w→0 solutions.
    
    Args:
        u_logits: [B, m] logits from neural network
        eps: small positive constant ε
        clamp: optional (min, max) bounds for numerical hygiene
        
    Returns:
        [B, m] positive weights with mean scale ~1
    """
    assert eps > 0, f"eps must be positive, got {eps}"
    
    # Map to positive values
    w_tilde = torch.exp(u_logits)  # [B, m]
    
    # Compute mean scale
    w_bar = w_tilde.mean(dim=1, keepdim=True)  # [B, 1]
    
    # Normalize to mean scale ~1 and add small offset
    w = eps + w_tilde / (w_bar + 1e-12)  # [B, m]
    
    # Optional clamping for numerical hygiene
    if clamp is not None:
        w_min, w_max = clamp
        if w_min is not None:
            w = torch.clamp(w, min=w_min)
        if w_max is not None:
            w = torch.clamp(w, max=w_max)
    
    return w


class WeightNet(nn.Module):
    """
    Neural network that outputs logits for weight parameterization.
    
    Simple MLP that takes features and outputs logits that get
    mapped to positive weights via map_logits_to_w.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, 
                 output_dim: int = None, dropout: float = 0.1):
        """
        Args:
            input_dim: dimension of input features
            hidden_dims: list of hidden layer dimensions
            output_dim: output dimension
            dropout: dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        if output_dim is None:
            raise ValueError("output_dim must be specified for WeightNet")
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs logits
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] input features
            
        Returns:
            [B, output_dim] logits for weight parameterization
        """
        return self.network(x)

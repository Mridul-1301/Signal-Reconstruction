"""
Linear operators for ECG signal imputation.

Implements the core linear operators:
- D: Second difference operator with stencil [1,-2,1]
- D^T: Adjoint transpose convolution
- A: A = D S₂ (second differences of unknowns only)
- A^T: A^T = S₂^T D^T (extract unknown positions)
"""

import torch


def apply_D(x: torch.Tensor) -> torch.Tensor:
    """
    Apply second difference operator D with stencil [1,-2,1].
    
    Args:
        x: [B, n] input signal
        
    Returns:
        [B, n-2] second differences
    """
    _, n = x.shape
    if n < 3:
        raise ValueError(f"Signal length {n} must be at least 3 for second differences")
    
    # Second difference: x[i+1] - 2*x[i] + x[i-1] for i in [1, n-2]
    x_shifted_right = x[:, 2:]  # x[i+1] for i in [1, n-2]
    x_center = x[:, 1:-1]       # x[i] for i in [1, n-2]
    x_shifted_left = x[:, :-2]  # x[i-1] for i in [1, n-2]
    
    return x_shifted_right - 2 * x_center + x_shifted_left


def apply_DT(y: torch.Tensor, n: int) -> torch.Tensor:
    """
    Apply adjoint of second difference operator D^T.
    
    Args:
        y: [B, n-2] input (second differences)
        n: target output length
        
    Returns:
        [B, n] adjoint result
    """
    B, m = y.shape
    if m != n - 2:
        raise ValueError(f"Input length {m} must equal n-2 = {n-2}")
    
    # Initialize output
    result = torch.zeros(B, n, device=y.device, dtype=y.dtype)
    
    # Apply transpose convolution: for each y[i], add to result at positions i, i+1, i+2
    # with coefficients 1, -2, 1 respectively
    for i in range(m):
        result[:, i] += y[:, i]      # coefficient 1
        result[:, i+1] += -2 * y[:, i]  # coefficient -2
        result[:, i+2] += y[:, i]    # coefficient 1
    
    return result


def A_apply(v: torch.Tensor, unknown_idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    Apply operator A = D S₂ where S₂ selects unknown positions.
    
    Args:
        v: [B, n_miss] unknown values
        unknown_idx: [n_miss] indices of unknown positions
        n: total signal length
        
    Returns:
        [B, m] second differences of unknowns, where m = n - 2
    """
    B, n_miss = v.shape
    
    # Create full signal with zeros for known positions
    x_full = torch.zeros(B, n, device=v.device, dtype=v.dtype)
    x_full[:, unknown_idx] = v
    
    # Apply second difference operator
    return apply_D(x_full)


def AT_apply(y: torch.Tensor, unknown_idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    Apply adjoint operator A^T = S₂^T D^T.
    
    Args:
        y: [B, m] input (second differences), where m = n - 2
        unknown_idx: [n_miss] indices of unknown positions
        n: total signal length
        
    Returns:
        [B, n_miss] result at unknown positions only
    """
    # Apply D^T to get full signal
    x_full = apply_DT(y, n)  # [B, n]
    
    # Extract unknown positions
    return x_full[:, unknown_idx]  # [B, n_miss]


def make_known_full(x: torch.Tensor, unknown_idx: torch.Tensor) -> torch.Tensor:
    """
    Create a full signal with known values and zeros for unknown positions.
    
    Args:
        x: [B, n] signal with NaNs for unknown values
        unknown_idx: [n_miss] indices of unknown positions
        
    Returns:
        [B, n] signal with zeros at unknown positions
    """
    x_known = x.clone()
    x_known[:, unknown_idx] = 0.0
    return x_known

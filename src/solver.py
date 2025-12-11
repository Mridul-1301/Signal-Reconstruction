"""
Conjugate Gradient solver and matrix-vector operations for the normal equations.

Implements K·z and h computation for the system K(w) v = h(w) where:
K(w) = A^T W^2 A, h(w) = A^T W^2 b
"""

import torch
from typing import Callable, Dict, Tuple
from .ops import A_apply, AT_apply


def matvec_K(z: torch.Tensor, w: torch.Tensor, unknown_idx: torch.Tensor, n: int, 
             jitter: float = 0.0) -> torch.Tensor:
    """
    Matrix-vector product K·z where K = A^T W^2 A.
    
    Args:
        z: [B, n_miss] input vector
        w: [B, m] weights (diagonal of W)
        unknown_idx: [n_miss] indices of unknown positions
        n: total signal length
        jitter: optional numerical jitter δ for stability
        
    Returns:
        [B, n_miss] result of K·z
    """
    # K z = A^T W^2 (A z) + δ z
    Az = A_apply(z, unknown_idx, n)  # [B, m]
    W2A = (w * w) * Az  # [B, m] - elementwise multiplication by W^2
    ATW = AT_apply(W2A, unknown_idx, n)  # [B, n_miss]
    
    if jitter != 0.0:
        ATW = ATW + jitter * z
    
    return ATW


def rhs_h(w: torch.Tensor, b: torch.Tensor, unknown_idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute right-hand side h = A^T W^2 b.
    
    Args:
        w: [B, m] weights (diagonal of W)
        b: [B, m] right-hand side vector
        unknown_idx: [n_miss] indices of unknown positions
        n: total signal length
        
    Returns:
        [B, n_miss] right-hand side vector
    """
    W2b = (w * w) * b  # [B, m] - elementwise multiplication by W^2
    return AT_apply(W2b, unknown_idx, n)  # [B, n_miss]


def cg(matvec_func: Callable, b: torch.Tensor, x0: torch.Tensor, 
       tol: float = 1e-6, maxiter: int = 2000, verbose: bool = False) -> Tuple[torch.Tensor, Dict]:
    """
    Conjugate Gradient solver for Ax = b.
    
    Args:
        matvec_func: function that computes A·x
        b: [n] right-hand side vector
        x0: [n] initial guess
        tol: relative residual tolerance
        maxiter: maximum number of iterations
        verbose: whether to print iteration info
        
    Returns:
        (x, info) where x is the solution and info contains convergence details
    """
    x = x0.clone()
    r = b - matvec_func(x)  # initial residual
    p = r.clone()  # initial search direction
    
    # Compute initial residual norm
    b_norm = torch.norm(b)
    if b_norm < 1e-14:
        # b is essentially zero, return zero solution
        return x, {"iters": [0], "relres": [0.0], "converged": True}
    
    relres_0 = torch.norm(r) / b_norm
    relres = relres_0
    
    iters = []
    relres_history = [relres_0.item()]
    
    if verbose:
        print(f"CG: iter 0, relres = {relres:.2e}")
    
    for k in range(maxiter):
        # Check convergence
        if relres <= tol:
            break
            
        # Compute Ap
        Ap = matvec_func(p)
        
        # Compute step size
        pAp = torch.dot(p, Ap)
        if pAp <= 0:
            if verbose:
                print(f"CG: Warning - p^T A p = {pAp:.2e} <= 0 at iter {k}")
            break
            
        alpha = torch.dot(r, r) / pAp
        
        # Update solution and residual
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        # Compute new residual norm
        relres = torch.norm(r_new) / b_norm
        relres_history.append(relres.item())
        
        # Check for convergence
        if relres <= tol:
            iters.append(k + 1)
            break
            
        # Compute new search direction
        beta = torch.dot(r_new, r_new) / torch.dot(r, r)
        p = r_new + beta * p
        r = r_new
        
        iters.append(k + 1)
        
        if verbose and (k + 1) % 100 == 0:
            print(f"CG: iter {k+1}, relres = {relres:.2e}")
    
    converged = relres <= tol
    
    if verbose:
        if converged:
            print(f"CG: converged in {len(iters)} iterations, final relres = {relres:.2e}")
        else:
            print(f"CG: did not converge after {maxiter} iterations, final relres = {relres:.2e}")
    
    info = {
        "iters": iters,
        "relres": relres_history,
        "converged": converged,
        "final_relres": relres.item()
    }
    
    return x, info


def solve_forward(w: torch.Tensor, b: torch.Tensor, unknown_idx: torch.Tensor, n: int,
                  cg_config) -> Tuple[torch.Tensor, Dict]:
    """
    Solve the forward system K(w) v = h(w) using CG.
    
    Args:
        w: [B, m] weights
        b: [B, m] right-hand side vector  
        unknown_idx: [n_miss] indices of unknown positions
        n: total signal length
        cg_config: CG configuration
        
    Returns:
        (v, info) where v is the solution and info contains convergence details
    """
    B, _ = w.shape
    n_miss = len(unknown_idx)
    
    # Compute right-hand side
    h = rhs_h(w, b, unknown_idx, n)  # [B, n_miss]
    
    # Solve per batch item for stability
    v_list = []
    info_list = []
    
    for i in range(B):
        # Create matrix-vector function for this batch item
        w_i = w[i:i+1]  # Capture the weight slice
        def K_mv(z):
            return matvec_K(z.unsqueeze(0), w_i, unknown_idx, n, jitter=cg_config.jitter).squeeze(0)
        
        v0 = torch.zeros(n_miss, device=w.device, dtype=w.dtype)
        vi, infoi = cg(K_mv, h[i], v0, 
                      tol=cg_config.tol, maxiter=cg_config.maxiter,
                      verbose=cg_config.verbose)
        v_list.append(vi)
        info_list.append(infoi)
    
    v = torch.stack(v_list, dim=0)  # [B, n_miss]
    
    # Aggregate info
    info = {
        "iters": [inf["iters"][-1] if inf["iters"] else 0 for inf in info_list],
        "relres": [inf["final_relres"] for inf in info_list],
        "converged": [inf["converged"] for inf in info_list]
    }
    
    return v, info


def solve_adjoint(w: torch.Tensor, g: torch.Tensor, unknown_idx: torch.Tensor, n: int,
                  cg_config) -> Tuple[torch.Tensor, Dict]:
    """
    Solve the adjoint system K(w) y = g using CG.
    
    Args:
        w: [B, m] weights
        g: [B, n_miss] adjoint right-hand side
        unknown_idx: [n_miss] indices of unknown positions
        n: total signal length
        cg_config: CG configuration
        
    Returns:
        (y, info) where y is the solution and info contains convergence details
    """
    B, _ = w.shape
    n_miss = len(unknown_idx)
    
    # Solve per batch item for stability
    y_list = []
    info_list = []
    
    for i in range(B):
        # Create matrix-vector function for this batch item
        w_i = w[i:i+1]  # Capture the weight slice
        def K_mv(z):
            return matvec_K(z.unsqueeze(0), w_i, unknown_idx, n, jitter=cg_config.jitter).squeeze(0)
        
        y0 = torch.zeros(n_miss, device=w.device, dtype=w.dtype)
        yi, infoi = cg(K_mv, g[i], y0,
                      tol=cg_config.tol, maxiter=cg_config.maxiter,
                      verbose=cg_config.verbose)
        y_list.append(yi)
        info_list.append(infoi)
    
    y = torch.stack(y_list, dim=0)  # [B, n_miss]
    
    # Aggregate info
    info = {
        "iters": [inf["iters"][-1] if inf["iters"] else 0 for inf in info_list],
        "relres": [inf["final_relres"] for inf in info_list],
        "converged": [inf["converged"] for inf in info_list]
    }
    
    return y, info

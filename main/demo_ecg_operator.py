#!/usr/bin/env python3
"""
Clean Demo Script for ECG Signal Imputation Operator (implicit gradient only).

Demonstrates the implicit gradient pathway with an outer loss that depends on
the optimizer solution v*.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_path)

from src.config import CGConfig, LayerConfig
from src.layer import DiagOpLayer
from src.weights import WeightNet


def load_ecg_data():
    """Load real ECG data from file."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ecg_missing.txt')
    
    # Read the data file
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse the data
    values = []
    for line in lines:
        line = line.strip()
        if line == "NaN":
            values.append(float('nan'))
        else:
            values.append(float(line))
    
    # Convert to tensor
    x_full = torch.tensor(values, dtype=torch.float32)
    
    # Find unknown positions (where values are NaN)
    unknown_idx = torch.where(torch.isnan(x_full))[0]
    
    # Create true signal by interpolating missing values for comparison
    x_true = x_full.clone()
    known_mask = ~torch.isnan(x_full)
    if known_mask.sum() > 0:
        # Simple linear interpolation for missing values using numpy
        known_indices = torch.where(known_mask)[0].numpy()
        known_values = x_full[known_indices].numpy()
        missing_indices = torch.where(~known_mask)[0].numpy()
        
        if len(missing_indices) > 0:
            interpolated_values = np.interp(missing_indices, known_indices, known_values)
            x_true[~known_mask] = torch.tensor(interpolated_values, dtype=torch.float32)
    
    return x_full, unknown_idx, x_true


def create_features(x_full, unknown_idx):
    """Create features for the weight network."""
    x_clean = torch.nan_to_num(x_full, nan=0.0)
    mean_val = x_clean.mean()
    std_val = x_clean.std()
    pos_features = torch.zeros(len(x_full))
    pos_features[unknown_idx] = 1.0
    feats = torch.cat([mean_val.unsqueeze(0), std_val.unsqueeze(0), pos_features])
    return feats


def train_model(layer, weight_net, x_full, unknown_idx, epochs=50):
    """Train the model using implicit gradient."""
    print("=== Implicit Mode Training ===")
    
    n = len(x_full)
    m = n - 2
    
    # Training setup
    optimizer = torch.optim.Adam(weight_net.parameters(), lr=1e-3)
    losses = []
    
    print(f"Signal length: {n}")
    print(f"Missing values: {len(unknown_idx)} at positions {unknown_idx.tolist()}")
    print(f"Weight dimension: {m}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Generate features
        feats = create_features(x_full, unknown_idx)
        
        # Forward pass
        u_logits = weight_net(feats.unsqueeze(0))
        out = layer(u_logits, x_full.unsqueeze(0), unknown_idx)
        
        # Implicit mode: L = φ(w) + 0.5 * ||v||^2
        phi_loss = out["phi"].sum()
        v_loss = 0.5 * torch.sum(out["v"] ** 2)
        loss = phi_loss + v_loss
        
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f} (φ = {phi_loss.item():.6f}, L(v) = {v_loss.item():.6f})")
    
    # Reconstruct signal
    with torch.no_grad():
        feats = create_features(x_full, unknown_idx)
        u_logits = weight_net(feats.unsqueeze(0))
        out = layer(u_logits, x_full.unsqueeze(0), unknown_idx)
        v = out["v"].squeeze(0)
        
        # Create full reconstruction
        x_reconstructed = x_full.clone()
        x_reconstructed[unknown_idx] = v
    
    print(f"Final loss: {losses[-1]:.6f}")
    
    return x_reconstructed, losses, weight_net


def plot_results(x_true, x_full, x_reconstructed, losses, mode):
    """Plot the results."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot signal
    t = torch.arange(len(x_true))
    ax1.plot(t, x_true, 'b-', label='True signal (interpolated)', linewidth=1.5, alpha=0.7)
    
    # Plot observed values (non-NaN)
    observed_mask = ~torch.isnan(x_full)
    ax1.plot(t[observed_mask], x_full[observed_mask], 'ro', label='Observed', markersize=4, alpha=0.8)
    
    # Plot reconstructed values at missing positions
    missing_mask = torch.isnan(x_full)
    ax1.plot(t[missing_mask], x_reconstructed[missing_mask], 'g.', label='Reconstructed', markersize=6, alpha=0.8)
    
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('ECG Amplitude')
    ax1.set_title(f'ECG Signal Reconstruction ({mode} mode)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(losses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Training Loss ({mode} mode)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'demo_{mode.lower()}_mode.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main demo function (implicit gradient only)."""
    print("ECG Signal Imputation Operator Demo")
    print("=" * 50)
    
    # Load and display ECG data info
    x_full, unknown_idx, x_true = load_ecg_data()
    n = len(x_full)
    n_known = (~torch.isnan(x_full)).sum().item()
    n_miss = len(unknown_idx)
    
    print("Real ECG Data Information:")
    print(f"  Total length: {n}")
    print(f"  Known values: {n_known}")
    print(f"  Missing values: {n_miss}")
    print(f"  Missing rate: {n_miss/n*100:.1f}%")
    print(f"  Signal range: [{x_full[~torch.isnan(x_full)].min():.4f}, {x_full[~torch.isnan(x_full)].max():.4f}]")
    print()
    
    # Create configurations
    cg_config = CGConfig(tol=1e-6, maxiter=1000, verbose=False)
    
    # Implicit mode only
    layer_config_impl = LayerConfig(mode="implicit", eps_w=1e-4)
    layer_impl = DiagOpLayer(cg_config, layer_config_impl)
    weight_net_impl = WeightNet(input_dim=n+2, hidden_dims=[32, 16], output_dim=n-2)
    
    x_reconstructed_impl, losses_impl, _ = train_model(layer_impl, weight_net_impl, x_full, unknown_idx)
    print(f"Implicit reconstruction error: {torch.norm(x_reconstructed_impl - x_true):.6f}")
    print()
    
    # Plot results
    try:
        plot_results(x_true, x_full, x_reconstructed_impl, losses_impl, "Implicit")
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    print("\nDemo completed successfully!")
    print("Key features demonstrated:")
    print("✓ Implicit mode: Adjoint gradient through optimizer")
    print("✓ Safe weight parameterization")
    print("✓ Conjugate gradient solver")
    print("✓ Linear operators with adjoint identity")


if __name__ == "__main__":
    main()

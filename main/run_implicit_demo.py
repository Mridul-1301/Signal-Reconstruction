#!/usr/bin/env python3
import os, sys, torch, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import CGConfig, LayerConfig
from src.layer import DiagOpLayer
from src.weights import WeightNet

def load_ecg_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ecg_missing.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        vals = [float('nan') if (ln := line.strip()) == 'NaN' else float(ln) for line in f]
    x_full = torch.tensor(vals, dtype=torch.float32)
    unknown_idx = torch.where(torch.isnan(x_full))[0]
    x_true = x_full.clone()
    known = ~torch.isnan(x_full)
    if known.any():
        ki = torch.where(known)[0].numpy(); kv = x_full[ki].numpy(); mi = torch.where(~known)[0].numpy()
        if len(mi) > 0:
            x_true[~known] = torch.tensor(np.interp(mi, ki, kv), dtype=torch.float32)
    return x_full, unknown_idx, x_true

def create_features(x_full, unknown_idx):
    x_clean = torch.nan_to_num(x_full, nan=0.0)
    mean_val = x_clean.mean(); std_val = x_clean.std()
    pos = torch.zeros(len(x_full)); pos[unknown_idx] = 1.0
    return torch.cat([mean_val.unsqueeze(0), std_val.unsqueeze(0), pos])

def main():
    x_full, unknown_idx, x_true = load_ecg_data()
    n = len(x_full); m = n - 2
    layer = DiagOpLayer(CGConfig(tol=1e-6, maxiter=1000, verbose=False), LayerConfig(mode='implicit', eps_w=1e-4))
    net = WeightNet(input_dim=n+2, hidden_dims=[32,16], output_dim=m)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for epoch in range(50):
        opt.zero_grad(); feats = create_features(x_full, unknown_idx); u = net(feats.unsqueeze(0))
        out = layer(u, x_full.unsqueeze(0), unknown_idx)
        phi = out['phi'].sum(); v_loss = 0.5 * torch.sum(out['v']**2); loss = phi + v_loss
        losses.append(loss.item()); loss.backward(); opt.step()
        if (epoch+1)%10==0:
            print(f'Epoch {epoch+1}: loss={loss.item():.6f} (phi={phi.item():.6f}, v2={v_loss.item():.6f})')
    with torch.no_grad():
        feats = create_features(x_full, unknown_idx); u = net(feats.unsqueeze(0)); out = layer(u, x_full.unsqueeze(0), unknown_idx)
        v = out['v'].squeeze(0); x_rec = x_full.clone(); x_rec[unknown_idx] = v
    err = torch.norm(x_rec - x_true).item(); print(f'Implicit reconstruction error: {err:.6f}')
    import torch as _t
    t = _t.arange(len(x_true)); fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
    ax1.plot(t, x_true, 'b-', label='True', alpha=0.7); obs = ~_t.isnan(x_full)
    ax1.plot(t[obs], x_full[obs], 'ro', ms=4, alpha=0.8, label='Observed'); miss = _t.isnan(x_full)
    ax1.plot(t[miss], x_rec[miss], 'g.', ms=6, alpha=0.8, label='Reconstructed'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(losses); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); outp = os.path.join(os.path.dirname(__file__), 'run_implicit_demo.png'); plt.savefig(outp, dpi=150, bbox_inches='tight'); print('Saved plot:', outp)
if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        residual = x
        x = self.inp(x)
        x = self.activation(x)
        x = self.out(x)
        return residual + x

class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)

print("Loading pieces...")
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"pieces/piece_{i}.pth", map_location='cpu', weights_only=True)

# Categorize by weight matrix shape
inps, outs = [], []
final_idx = -1

for idx, sd in pieces.items():
    w_shape = sd['weight'].shape
    if w_shape == (1, 48):
        final_idx = idx
    elif w_shape == (96, 48):
        inps.append(idx)
    elif w_shape == (48, 96):
        outs.append(idx)

print(f"Found {len(inps)} inputs, {len(outs)} outputs, and final layer at index {final_idx}")

print("Matching blocks via W_out @ W_in negative diagonals...")
cost_matrix = np.zeros((48, 48))
for i, inp_idx in enumerate(inps):
    W_in = pieces[inp_idx]['weight']
    for j, out_idx in enumerate(outs):
        W_out = pieces[out_idx]['weight']
        W_prod = W_out @ W_in
        cost_matrix[i, j] = torch.diag(W_prod).mean().item()

row_ind, col_ind = linear_sum_assignment(cost_matrix)
paired_blocks = [(inps[r], outs[c]) for r, c in zip(row_ind, col_ind)]

print("Loading historical_data.csv...")
df = pd.read_csv('historical_data.csv')
X_cols = [col for col in df.columns if col.startswith('measurement_')]
X = torch.tensor(df[X_cols].values, dtype=torch.float32)
y_target = torch.tensor(df['pred'].values, dtype=torch.float32).unsqueeze(1)

blocks = []
for inp_idx, out_idx in paired_blocks:
    b = Block(48, 96)
    sd = {
        'inp.weight': pieces[inp_idx]['weight'],
        'inp.bias': pieces[inp_idx]['bias'],
        'out.weight': pieces[out_idx]['weight'],
        'out.bias': pieces[out_idx]['bias']
    }
    b.load_state_dict(sd)
    b.eval()
    
    with torch.no_grad():
        out_x = b(X)
        delta = torch.norm(out_x - X, dim=1).mean().item()
        
    blocks.append({'inp_idx': inp_idx, 'out_idx': out_idx, 'module': b, 'delta': delta})

blocks.sort(key=lambda b: b['delta'])
current_order = blocks.copy()

final_module = LastLayer(48, 1)
final_sd = {
        'layer.weight': pieces[final_idx]['weight'],
        'layer.bias': pieces[final_idx]['bias']
    }
final_module.load_state_dict(final_sd)
final_module.eval()

def evaluate_order(order):
    with torch.no_grad():
        curr_X = X
        for b in order:
            curr_X = b['module'](curr_X)
        preds = final_module(curr_X)
        return torch.nn.functional.mse_loss(preds, y_target).item()

best_mse = evaluate_order(current_order)
print(f"Initial Sorted MSE: {best_mse:.6f}")

print("Beginning local search to find perfect sequence...")
improved = True
while improved and best_mse > 1e-6:
    improved = False
    for i in range(48):
        for j in range(i + 1, 48):
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            mse = evaluate_order(new_order)
            
            if mse < best_mse:
                best_mse = mse
                current_order = new_order
                improved = True
                print(f"Found better sequence! Swapped {i} & {j}. New MSE: {best_mse:.8f}")

permutation = []
for b in current_order:
    permutation.extend([b['inp_idx'], b['out_idx']])
permutation.append(final_idx)

print("\n--- FINAL PERMUTATION SOLUTION ---")
print(",".join(map(str, permutation)))
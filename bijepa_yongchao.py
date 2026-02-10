# -*- coding: utf-8 -*-
# @author:yongchao.huang@abdn.ac.uk

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Sine waves.

## shared wave generation and splitting.
"""

import torch
import numpy as np
import random
import os

# =============================================================================
# 0. Reproducibility Setup
# =============================================================================
def set_seed(seed=42):
    """Sets the seed for reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to: {seed}")

set_seed(42)  # Locking the seed!

# =============================================================================
# 1. Dataset Generation Logic
# =============================================================================
def generate_sine_waves(batch_size, seq_len=20):
    """Generates sine waves."""
    x = np.linspace(0, 4*np.pi, seq_len)
    waves = []
    for _ in range(batch_size):
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)
        wave = np.sin(freq * x + phase)
        # Add tiny noise
        wave += np.random.normal(0, 0.05, seq_len)
        waves.append(wave)
    return torch.tensor(np.array(waves), dtype=torch.float32).unsqueeze(-1)

# =============================================================================
# 2. Generate and Save All Datasets
# =============================================================================
print("Generating Shared Datasets...")

# 1. Training Data (Enough for 2000 steps * 64 batch size = 128,000 samples)
# We generate a large pool to sample from, or just generate on the fly with fixed seed.
# However, to be 100% sure, let's pre-generate the specific batches used for plotting.

# A. Demo Data (For Initial Visualization)
demo_data = generate_sine_waves(batch_size=3, seq_len=20)

# B. Probe Training Data (Phase 2a/2b)
probe_data = generate_sine_waves(batch_size=1000, seq_len=20)

# C. Test Data (For Final Visualization Plot 2 & 3)
test_data = generate_sine_waves(batch_size=20, seq_len=20)

# Save to disk
torch.save({
    'demo_data': demo_data,
    'probe_data': probe_data,
    'test_data': test_data
}, 'sine_wave_data.pt')

print("Data saved to 'sine_wave_data.pt'")

"""## classic JEPA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from matplotlib.ticker import MaxNLocator
import os

# =============================================================================
# 0. Load Shared Data (Ensure fair comparison)
# =============================================================================
if not os.path.exists('sine_wave_data.pt'):
    raise FileNotFoundError("Run the data generation script first!")

shared_data = torch.load('sine_wave_data.pt')
demo_data = shared_data['demo_data']
probe_data = shared_data['probe_data']
test_data = shared_data['test_data']

# Helper to regenerate identical training batches
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_sine_waves(batch_size, seq_len=20):
    x = np.linspace(0, 4*np.pi, seq_len)
    waves = []
    for _ in range(batch_size):
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)
        wave = np.sin(freq * x + phase)
        wave += np.random.normal(0, 0.05, seq_len)
        waves.append(wave)
    return torch.tensor(np.array(waves), dtype=torch.float32).unsqueeze(-1)

# --- VISUALIZATION 0: Initial Data Samples ---
t_steps = np.arange(20)
plt.figure(figsize=(10, 4))
colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(t_steps, demo_data[i].flatten(), marker='o', markersize=4,
             label=f'Sample {i}', color=colors[i], alpha=0.7)
plt.title("Dataset Examples (3 Random Samples)", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 2. Model Components (Classic JEPA - Expressive Config)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_flat = x.squeeze(-1)
        out = self.net(x_flat)
        # No L2 Norm (Expressive Config)
        return out

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class ClassicJEPA(nn.Module):
    def __init__(self, input_len, embed_dim=16):
        super().__init__()
        # 1. Online Encoder (Context)
        self.online_encoder = Encoder(input_len, 64, embed_dim)

        # 2. Predictor (Forward Only: Context -> Target)
        self.predictor = Predictor(embed_dim, 64)

        # 3. Target Encoder (Target) - EMA Updated
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, momentum=0.995):
        with torch.no_grad():
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x_raw, y_raw):
        # 1. Encode Target (Stop Gradient)
        with torch.no_grad():
            target_embed_y = self.target_encoder(y_raw)

        # 2. Encode Context (Online)
        online_embed_x = self.online_encoder(x_raw)

        # 3. Predict Future (Forward Only)
        pred_y = self.predictor(online_embed_x)

        # 4. Loss (Uni-directional)
        loss = nn.functional.mse_loss(pred_y, target_embed_y)

        return loss

# =============================================================================
# 3. Execution Pipeline
# =============================================================================
seq_len = 20
split_point = 10
embed_dim = 16

# RESET SEED
set_seed(42)

# Init Classic JEPA
jepa_model = ClassicJEPA(input_len=split_point, embed_dim=embed_dim)
# Expressive Config: AdamW with Weight Decay
optimizer = optim.AdamW(jepa_model.parameters(), lr=1e-3, weight_decay=1e-4)

losses = []

# --- PHASE 1: Pre-training ---
print("Phase 1: Self-Supervised Pre-training (Classic JEPA)...")
for step in range(2000):
    data = generate_sine_waves(batch_size=64, seq_len=seq_len)
    x_raw = data[:, :split_point, :]
    y_raw = data[:, split_point:, :]

    optimizer.zero_grad()
    loss = jepa_model(x_raw, y_raw)
    loss.backward()
    optimizer.step()

    jepa_model.update_target_encoder()

    losses.append(loss.item())

    if step % 500 == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

# Freeze Model
jepa_model.eval()
for p in jepa_model.parameters(): p.requires_grad = False

# --- PHASE 2a: Protocol A (Encoder Probe) ---
print("\nPhase 2a: Protocol A (Encoder Probe)...")
train_x = probe_data[:, :split_point, :]
train_y_target = probe_data[:, split_point, :]

probe_A = nn.Linear(embed_dim, 1)
opt_A = optim.Adam(probe_A.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
    preds = probe_A(s_x)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_A.zero_grad(); loss.backward(); opt_A.step()
print(f"Protocol A (Encoder) MSE: {loss.item():.5f}")

# --- PHASE 2b: Protocol B (Predictor Probe) ---
print("Phase 2b: Protocol B (Generative Forecasting)...")
probe_B = nn.Linear(embed_dim, 1)
opt_B = optim.Adam(probe_B.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
        s_y_hat = jepa_model.predictor(s_x) # Use the single predictor
    preds = probe_B(s_y_hat)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_B.zero_grad(); loss.backward(); opt_B.step()
print(f"Protocol B (Predictor) MSE: {loss.item():.5f}")

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Training Stability
plt.subplot(1, 3, 1)
plt.plot(losses, label='JEPA Loss', linewidth=2, color='tab:blue')
plt.title("Training Stability (Classic JEPA)", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Plot 2: Forecast Accuracy
test_x = test_data[:, :split_point, :]
test_y_true = test_data[:, split_point, 0]

with torch.no_grad():
    s_x = jepa_model.online_encoder(test_x)
    s_y_hat = jepa_model.predictor(s_x)
    pred_A = probe_A(s_x).flatten()
    pred_B = probe_B(s_y_hat).flatten()

plt.subplot(1, 3, 2)
plt.plot(test_y_true, 'o', label='True Value (t=11)', color='black', alpha=0.3, markersize=8)
plt.plot(pred_A, 'x', label='Proto A (Encoder)', color='red', markersize=8, markeredgewidth=2)
plt.plot(pred_B, '^', label='Proto B (Predictor)', color='green', markersize=8, markeredgewidth=2)
plt.vlines(x=range(len(test_y_true)), ymin=test_y_true, ymax=pred_B, colors='green', linestyles='dotted', alpha=0.6, label='Error (Proto B)')
plt.title("Forecast Accuracy (Classic JEPA)", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(-1.2, 1.3)

# Plot 3: Single Sample
plt.subplot(1, 3, 3)
sample_idx = 0
sample_ctx = test_x[sample_idx].flatten().numpy()
sample_true_future = test_y_true[sample_idx].item()
sample_A = pred_A[sample_idx].item()
sample_B = pred_B[sample_idx].item()
t_ctx = np.arange(10)
t_fut = 10

plt.plot(t_ctx, sample_ctx, 'b-o', label='Context Input', linewidth=2)
plt.plot(t_fut, sample_true_future, 'o', color='black', alpha=0.3, markersize=10, label='True Future (t=11)')
plt.plot(t_fut, sample_A, 'rx', label='Proto A Forecast', markersize=10, markeredgewidth=2)
plt.plot(t_fut, sample_B, 'g^', label='Proto B Forecast', markersize=10, markeredgewidth=2)
plt.plot([9, 10], [sample_ctx[-1], sample_B], 'g--', alpha=0.5)
plt.title(f"Single Sample Forecast (Sample {sample_idx})", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-1.2, 1.3)

plt.tight_layout()
plt.show()

"""## Case 1-1: without L2-reg, embedding explosion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from matplotlib.ticker import MaxNLocator

# =============================================================================
# 0. Load Shared Data
# =============================================================================
if not os.path.exists('sine_wave_data.pt'):
    raise FileNotFoundError("Run the data generation script first!")

shared_data = torch.load('sine_wave_data.pt')
demo_data = shared_data['demo_data']
probe_data = shared_data['probe_data']
test_data = shared_data['test_data']

# Helper for on-the-fly training data (keep this for the main loop as it's huge)
# But we re-set seed to ensure the training batches are identical across scripts
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_sine_waves(batch_size, seq_len=20):
    """Re-defined locally just for the training loop generation"""
    x = np.linspace(0, 4*np.pi, seq_len)
    waves = []
    for _ in range(batch_size):
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)
        wave = np.sin(freq * x + phase)
        wave += np.random.normal(0, 0.05, seq_len)
        waves.append(wave)
    return torch.tensor(np.array(waves), dtype=torch.float32).unsqueeze(-1)

# --- VISUALIZATION 0: Initial Data Samples ---
t_steps = np.arange(20)
plt.figure(figsize=(10, 4))
colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(t_steps, demo_data[i].flatten(), marker='o', markersize=4,
             label=f'Sample {i}', color=colors[i], alpha=0.7)
plt.title("Dataset Examples (3 Random Samples)", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 2. Model Components (UNSTABLE: No LayerNorm, No L2 Norm)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),  <-- REMOVED
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),  <-- REMOVED
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_flat = x.squeeze(-1)
        out = self.net(x_flat)
        # return F.normalize(out, dim=-1) <-- REMOVED L2 NORM
        return out # Raw output

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), <-- REMOVED
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class BiJEPA(nn.Module):
    def __init__(self, input_len, embed_dim=16):
        super().__init__()
        self.online_encoder = Encoder(input_len, 64, embed_dim)
        self.fwd_predictor = Predictor(embed_dim, 64) # X -> Y
        self.bwd_predictor = Predictor(embed_dim, 64) # Y -> X

        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, momentum=0.995):
        with torch.no_grad():
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x_raw, y_raw):
        with torch.no_grad():
            target_embed_x = self.target_encoder(x_raw)
            target_embed_y = self.target_encoder(y_raw)
        online_embed_x = self.online_encoder(x_raw)
        online_embed_y = self.online_encoder(y_raw)
        pred_y = self.fwd_predictor(online_embed_x)
        pred_x = self.bwd_predictor(online_embed_y)
        loss_fwd = nn.functional.mse_loss(pred_y, target_embed_y)
        loss_bwd = nn.functional.mse_loss(pred_x, target_embed_x)
        return loss_fwd, loss_bwd

# =============================================================================
# 3. Execution Pipeline
# =============================================================================
seq_len = 20
split_point = 10
embed_dim = 16

# RESET SEED FOR TRAINING LOOP TO BE IDENTICAL
set_seed(42)

jepa_model = BiJEPA(input_len=split_point, embed_dim=embed_dim)
# CRITICAL CHANGE: Remove weight_decay
optimizer = optim.AdamW(jepa_model.parameters(), lr=1e-3, weight_decay=0.0)

losses = []

# --- PHASE 1: Pre-training ---
print("Phase 1: Self-Supervised Pre-training (Bi-JEPA)...")
for step in range(2000):
    data = generate_sine_waves(batch_size=64, seq_len=seq_len)
    x_raw = data[:, :split_point, :]
    y_raw = data[:, split_point:, :]

    optimizer.zero_grad()
    loss_fwd, loss_bwd = jepa_model(x_raw, y_raw)
    (loss_fwd + loss_bwd).backward()
    optimizer.step()
    jepa_model.update_target_encoder()

    losses.append((loss_fwd.item(), loss_bwd.item()))

    if step % 500 == 0:
        print(f"Step {step}: Loss Fwd {loss_fwd.item():.4f}")

# Freeze Model
jepa_model.eval()
for p in jepa_model.parameters(): p.requires_grad = False

# --- PHASE 2a: Protocol A ---
print("\nPhase 2a: Protocol A (Encoder Probe)...")
# USE SHARED PROBE DATA
train_x = probe_data[:, :split_point, :]
train_y_target = probe_data[:, split_point, :]

probe_A = nn.Linear(embed_dim, 1)
opt_A = optim.Adam(probe_A.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
    preds = probe_A(s_x)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_A.zero_grad(); loss.backward(); opt_A.step()
print(f"Protocol A (Encoder) MSE: {loss.item():.5f}")

# --- PHASE 2b: Protocol B ---
print("Phase 2b: Protocol B (Generative Forecasting)...")
probe_B = nn.Linear(embed_dim, 1)
opt_B = optim.Adam(probe_B.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
        s_y_hat = jepa_model.fwd_predictor(s_x)
    preds = probe_B(s_y_hat)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_B.zero_grad(); loss.backward(); opt_B.step()
print(f"Protocol B (Predictor) MSE: {loss.item():.5f}")

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Training Stability
plt.subplot(1, 3, 1)
f_loss = [l[0] for l in losses]
b_loss = [l[1] for l in losses]
plt.plot(f_loss, label='Forward Loss', linewidth=2)
plt.plot(b_loss, label='Backward Loss', linewidth=2)
plt.title("Training Stability (UNSTABLE: No Norm/Decay)", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Plot 2: Forecast Accuracy
# USE SHARED TEST DATA
test_x = test_data[:, :split_point, :]
test_y_true = test_data[:, split_point, 0]

with torch.no_grad():
    s_x = jepa_model.online_encoder(test_x)
    s_y_hat = jepa_model.fwd_predictor(s_x)
    pred_A = probe_A(s_x).flatten()
    pred_B = probe_B(s_y_hat).flatten()

plt.subplot(1, 3, 2)
plt.plot(test_y_true, 'o', label='True Value (t=11)', color='black', alpha=0.3, markersize=8)
plt.plot(pred_A, 'x', label='Proto A (Encoder)', color='red', markersize=8, markeredgewidth=2)
plt.plot(pred_B, '^', label='Proto B (Predictor)', color='green', markersize=8, markeredgewidth=2)
plt.vlines(x=range(len(test_y_true)), ymin=test_y_true, ymax=pred_B, colors='green', linestyles='dotted', alpha=0.6, label='Error (Proto B)')
plt.title("Forecast Accuracy (Protocol A vs B)", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(-1.2, 1.3)

# Plot 3: Single Sample
plt.subplot(1, 3, 3)
sample_idx = 0
sample_ctx = test_x[sample_idx].flatten().numpy()
sample_true_future = test_y_true[sample_idx].item()
sample_A = pred_A[sample_idx].item()
sample_B = pred_B[sample_idx].item()
t_ctx = np.arange(10)
t_fut = 10

plt.plot(t_ctx, sample_ctx, 'b-o', label='Context Input', linewidth=2)
plt.plot(t_fut, sample_true_future, 'o', color='black', alpha=0.3, markersize=10, label='True Future (t=11)')
plt.plot(t_fut, sample_A, 'rx', label='Proto A Forecast', markersize=10, markeredgewidth=2)
plt.plot(t_fut, sample_B, 'g^', label='Proto B Forecast', markersize=10, markeredgewidth=2)
plt.plot([9, 10], [sample_ctx[-1], sample_B], 'g--', alpha=0.5)
plt.title(f"Single Sample Forecast (Sample {sample_idx})", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-1.2, 1.3)

plt.tight_layout()
plt.show()

"""## Case 1-2: sine wave - expressive model

L2 reg removed, but still have:

Layer Normalization + Weight Decay (via AdamW)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from matplotlib.ticker import MaxNLocator
import os

# =============================================================================
# 0. Load Shared Data
# =============================================================================
if not os.path.exists('sine_wave_data.pt'):
    raise FileNotFoundError("Run the data generation script first!")

shared_data = torch.load('sine_wave_data.pt')
demo_data = shared_data['demo_data']
probe_data = shared_data['probe_data']
test_data = shared_data['test_data']

# Helper for on-the-fly training data
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_sine_waves(batch_size, seq_len=20):
    """Re-defined locally just for the training loop generation"""
    x = np.linspace(0, 4*np.pi, seq_len)
    waves = []
    for _ in range(batch_size):
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)
        wave = np.sin(freq * x + phase)
        wave += np.random.normal(0, 0.05, seq_len)
        waves.append(wave)
    return torch.tensor(np.array(waves), dtype=torch.float32).unsqueeze(-1)

# --- VISUALIZATION 0: Initial Data Samples ---
t_steps = np.arange(20)
plt.figure(figsize=(10, 4))
colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(t_steps, demo_data[i].flatten(), marker='o', markersize=4,
             label=f'Sample {i}', color=colors[i], alpha=0.7)
plt.title("Dataset Examples (3 Random Samples)", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 2. Model Components (EXPRESSIVE: LayerNorm + Decay)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_flat = x.squeeze(-1)
        out = self.net(x_flat)
        # --- NO L2 NORMALIZATION ---
        return out

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class BiJEPA(nn.Module):
    def __init__(self, input_len, embed_dim=16):
        super().__init__()
        self.online_encoder = Encoder(input_len, 64, embed_dim)
        self.fwd_predictor = Predictor(embed_dim, 64) # X -> Y
        self.bwd_predictor = Predictor(embed_dim, 64) # Y -> X

        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, momentum=0.995):
        with torch.no_grad():
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x_raw, y_raw):
        with torch.no_grad():
            target_embed_x = self.target_encoder(x_raw)
            target_embed_y = self.target_encoder(y_raw)
        online_embed_x = self.online_encoder(x_raw)
        online_embed_y = self.online_encoder(y_raw)
        pred_y = self.fwd_predictor(online_embed_x)
        pred_x = self.bwd_predictor(online_embed_y)
        loss_fwd = nn.functional.mse_loss(pred_y, target_embed_y)
        loss_bwd = nn.functional.mse_loss(pred_x, target_embed_x)
        return loss_fwd, loss_bwd

# =============================================================================
# 3. Execution Pipeline
# =============================================================================
seq_len = 20
split_point = 10
embed_dim = 16

# RESET SEED FOR TRAINING LOOP TO BE IDENTICAL
set_seed(42)

jepa_model = BiJEPA(input_len=split_point, embed_dim=embed_dim)
# ENABLE WEIGHT DECAY
optimizer = optim.AdamW(jepa_model.parameters(), lr=1e-3, weight_decay=1e-4)

losses = []

# --- PHASE 1: Pre-training ---
print("Phase 1: Self-Supervised Pre-training (Bi-JEPA)...")
for step in range(2000):
    data = generate_sine_waves(batch_size=64, seq_len=seq_len)
    x_raw = data[:, :split_point, :]
    y_raw = data[:, split_point:, :]

    optimizer.zero_grad()
    loss_fwd, loss_bwd = jepa_model(x_raw, y_raw)
    (loss_fwd + loss_bwd).backward()
    optimizer.step()
    jepa_model.update_target_encoder()

    losses.append((loss_fwd.item(), loss_bwd.item()))

    if step % 500 == 0:
        print(f"Step {step}: Loss Fwd {loss_fwd.item():.4f}")

# Freeze Model
jepa_model.eval()
for p in jepa_model.parameters(): p.requires_grad = False

# --- PHASE 2a: Protocol A ---
print("\nPhase 2a: Protocol A (Encoder Probe)...")
# USE SHARED PROBE DATA
train_x = probe_data[:, :split_point, :]
train_y_target = probe_data[:, split_point, :]

probe_A = nn.Linear(embed_dim, 1)
opt_A = optim.Adam(probe_A.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
    preds = probe_A(s_x)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_A.zero_grad(); loss.backward(); opt_A.step()
print(f"Protocol A (Encoder) MSE: {loss.item():.5f}")

# --- PHASE 2b: Protocol B ---
print("Phase 2b: Protocol B (Generative Forecasting)...")
probe_B = nn.Linear(embed_dim, 1)
opt_B = optim.Adam(probe_B.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
        s_y_hat = jepa_model.fwd_predictor(s_x)
    preds = probe_B(s_y_hat)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_B.zero_grad(); loss.backward(); opt_B.step()
print(f"Protocol B (Predictor) MSE: {loss.item():.5f}")

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Training Stability
plt.subplot(1, 3, 1)
f_loss = [l[0] for l in losses]
b_loss = [l[1] for l in losses]
plt.plot(f_loss, label='Forward Loss', linewidth=2)
plt.plot(b_loss, label='Backward Loss', linewidth=2)
plt.title("Training Stability (Expressive: LayerNorm/Decay)", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Plot 2: Forecast Accuracy
# USE SHARED TEST DATA
test_x = test_data[:, :split_point, :]
test_y_true = test_data[:, split_point, 0]

with torch.no_grad():
    s_x = jepa_model.online_encoder(test_x)
    s_y_hat = jepa_model.fwd_predictor(s_x)
    pred_A = probe_A(s_x).flatten()
    pred_B = probe_B(s_y_hat).flatten()

plt.subplot(1, 3, 2)
plt.plot(test_y_true, 'o', label='True Value (t=11)', color='black', alpha=0.3, markersize=8)
plt.plot(pred_A, 'x', label='Proto A (Encoder)', color='red', markersize=8, markeredgewidth=2)
plt.plot(pred_B, '^', label='Proto B (Predictor)', color='green', markersize=8, markeredgewidth=2)
plt.vlines(x=range(len(test_y_true)), ymin=test_y_true, ymax=pred_B, colors='green', linestyles='dotted', alpha=0.6, label='Error (Proto B)')
plt.title("Forecast Accuracy (Protocol A vs B)", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(-1.2, 1.3)

# Plot 3: Single Sample
plt.subplot(1, 3, 3)
sample_idx = 0
sample_ctx = test_x[sample_idx].flatten().numpy()
sample_true_future = test_y_true[sample_idx].item()
sample_A = pred_A[sample_idx].item()
sample_B = pred_B[sample_idx].item()
t_ctx = np.arange(10)
t_fut = 10

plt.plot(t_ctx, sample_ctx, 'b-o', label='Context Input', linewidth=2)
plt.plot(t_fut, sample_true_future, 'o', color='black', alpha=0.3, markersize=10, label='True Future (t=11)')
plt.plot(t_fut, sample_A, 'rx', label='Proto A Forecast', markersize=10, markeredgewidth=2)
plt.plot(t_fut, sample_B, 'g^', label='Proto B Forecast', markersize=10, markeredgewidth=2)
plt.plot([9, 10], [sample_ctx[-1], sample_B], 'g--', alpha=0.5)
plt.title(f"Single Sample Forecast (Sample {sample_idx})", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-1.2, 1.3)

plt.tight_layout()
plt.show()

"""## Case 1-3: sine wave - restrive model (L2 reg, embedding vector lives on a unit sphere)

## what improved:

```
Add Normalization: We will force the encoder outputs to be unit vectors using F.normalize.

Add Weight Decay: To further discourage exploding weights.

Add Evaluation (Linear Probe): we will freeze the trained encoder and train a simple linear regression model to see if the embeddings actually capture the sine wave's future.

```

## future potential improve:
 weighted fwd and bwd losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from matplotlib.ticker import MaxNLocator
import os

# =============================================================================
# 0. Load Shared Data
# =============================================================================
if not os.path.exists('sine_wave_data.pt'):
    raise FileNotFoundError("Run the data generation script first!")

shared_data = torch.load('sine_wave_data.pt')
demo_data = shared_data['demo_data']
probe_data = shared_data['probe_data']
test_data = shared_data['test_data']

# Helper for on-the-fly training data
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_sine_waves(batch_size, seq_len=20):
    """Re-defined locally just for the training loop generation"""
    x = np.linspace(0, 4*np.pi, seq_len)
    waves = []
    for _ in range(batch_size):
        freq = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2*np.pi)
        wave = np.sin(freq * x + phase)
        wave += np.random.normal(0, 0.05, seq_len)
        waves.append(wave)
    return torch.tensor(np.array(waves), dtype=torch.float32).unsqueeze(-1)

# --- VISUALIZATION 0: Initial Data Samples ---
t_steps = np.arange(20)
plt.figure(figsize=(10, 4))
colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(t_steps, demo_data[i].flatten(), marker='o', markersize=4,
             label=f'Sample {i}', color=colors[i], alpha=0.7)
plt.title("Dataset Examples (3 Random Samples)", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 2. Model Components (RESTRICTIVE: L2 Normalization)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_flat = x.squeeze(-1)
        out = self.net(x_flat)
        # --- HARD CONSTRAINT: L2 NORMALIZATION ---
        return F.normalize(out, dim=-1)

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class BiJEPA(nn.Module):
    def __init__(self, input_len, embed_dim=16):
        super().__init__()
        self.online_encoder = Encoder(input_len, 64, embed_dim)
        self.fwd_predictor = Predictor(embed_dim, 64) # X -> Y
        self.bwd_predictor = Predictor(embed_dim, 64) # Y -> X

        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, momentum=0.995):
        with torch.no_grad():
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x_raw, y_raw):
        with torch.no_grad():
            target_embed_x = self.target_encoder(x_raw)
            target_embed_y = self.target_encoder(y_raw)
        online_embed_x = self.online_encoder(x_raw)
        online_embed_y = self.online_encoder(y_raw)
        pred_y = self.fwd_predictor(online_embed_x)
        pred_x = self.bwd_predictor(online_embed_y)
        loss_fwd = nn.functional.mse_loss(pred_y, target_embed_y)
        loss_bwd = nn.functional.mse_loss(pred_x, target_embed_x)
        return loss_fwd, loss_bwd

# =============================================================================
# 3. Execution Pipeline
# =============================================================================
seq_len = 20
split_point = 10
embed_dim = 16

# RESET SEED FOR TRAINING LOOP TO BE IDENTICAL
set_seed(42)

jepa_model = BiJEPA(input_len=split_point, embed_dim=embed_dim)
# ENABLE WEIGHT DECAY
optimizer = optim.AdamW(jepa_model.parameters(), lr=1e-3, weight_decay=1e-4)

losses = []

# --- PHASE 1: Pre-training ---
print("Phase 1: Self-Supervised Pre-training (Bi-JEPA)...")
for step in range(2000):
    data = generate_sine_waves(batch_size=64, seq_len=seq_len)
    x_raw = data[:, :split_point, :]
    y_raw = data[:, split_point:, :]

    optimizer.zero_grad()
    loss_fwd, loss_bwd = jepa_model(x_raw, y_raw)
    (loss_fwd + loss_bwd).backward()
    optimizer.step()
    jepa_model.update_target_encoder()

    losses.append((loss_fwd.item(), loss_bwd.item()))

    if step % 500 == 0:
        print(f"Step {step}: Loss Fwd {loss_fwd.item():.4f}")

# Freeze Model
jepa_model.eval()
for p in jepa_model.parameters(): p.requires_grad = False

# --- PHASE 2a: Protocol A ---
print("\nPhase 2a: Protocol A (Encoder Probe)...")
# USE SHARED PROBE DATA
train_x = probe_data[:, :split_point, :]
train_y_target = probe_data[:, split_point, :]

probe_A = nn.Linear(embed_dim, 1)
opt_A = optim.Adam(probe_A.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
    preds = probe_A(s_x)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_A.zero_grad(); loss.backward(); opt_A.step()
print(f"Protocol A (Encoder) MSE: {loss.item():.5f}")

# --- PHASE 2b: Protocol B ---
print("Phase 2b: Protocol B (Generative Forecasting)...")
probe_B = nn.Linear(embed_dim, 1)
opt_B = optim.Adam(probe_B.parameters(), lr=0.01)

for epoch in range(100):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
        s_y_hat = jepa_model.fwd_predictor(s_x)
    preds = probe_B(s_y_hat)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_B.zero_grad(); loss.backward(); opt_B.step()
print(f"Protocol B (Predictor) MSE: {loss.item():.5f}")

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Training Stability
plt.subplot(1, 3, 1)
f_loss = [l[0] for l in losses]
b_loss = [l[1] for l in losses]
plt.plot(f_loss, label='Forward Loss', linewidth=2)
plt.plot(b_loss, label='Backward Loss', linewidth=2)
plt.title("Training Stability (Restrictive: Sphere Norm)", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Plot 2: Forecast Accuracy
# USE SHARED TEST DATA
test_x = test_data[:, :split_point, :]
test_y_true = test_data[:, split_point, 0]

with torch.no_grad():
    s_x = jepa_model.online_encoder(test_x)
    s_y_hat = jepa_model.fwd_predictor(s_x)
    pred_A = probe_A(s_x).flatten()
    pred_B = probe_B(s_y_hat).flatten()

plt.subplot(1, 3, 2)
plt.plot(test_y_true, 'o', label='True Value (t=11)', color='black', alpha=0.3, markersize=8)
plt.plot(pred_A, 'x', label='Proto A (Encoder)', color='red', markersize=8, markeredgewidth=2)
plt.plot(pred_B, '^', label='Proto B (Predictor)', color='green', markersize=8, markeredgewidth=2)
plt.vlines(x=range(len(test_y_true)), ymin=test_y_true, ymax=pred_B, colors='green', linestyles='dotted', alpha=0.6, label='Error (Proto B)')
plt.title("Forecast Accuracy (Protocol A vs B)", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(-1.2, 1.3)

# Plot 3: Single Sample
plt.subplot(1, 3, 3)
sample_idx = 0
sample_ctx = test_x[sample_idx].flatten().numpy()
sample_true_future = test_y_true[sample_idx].item()
sample_A = pred_A[sample_idx].item()
sample_B = pred_B[sample_idx].item()
t_ctx = np.arange(10)
t_fut = 10

plt.plot(t_ctx, sample_ctx, 'b-o', label='Context Input', linewidth=2)
plt.plot(t_fut, sample_true_future, 'o', color='black', alpha=0.3, markersize=10, label='True Future (t=11)')
plt.plot(t_fut, sample_A, 'rx', label='Proto A Forecast', markersize=10, markeredgewidth=2)
plt.plot(t_fut, sample_B, 'g^', label='Proto B Forecast', markersize=10, markeredgewidth=2)
plt.plot([9, 10], [sample_ctx[-1], sample_B], 'g--', alpha=0.5)
plt.title(f"Single Sample Forecast (Sample {sample_idx})", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-1.2, 1.3)

plt.tight_layout()
plt.show()

"""## Result notes:

### successfully implemented a Self-Supervised Learning (SSL) system that:

    Encodes a time series into a compressed abstract space (16-dim).

    Predicts the future representation from the past (Forward JEPA).

    Predicts the past representation from the future (Backward JEPA).

    Avoids Collapse using an EMA Target Encoder.

    Avoids Explosion using L2 Normalization.

 Bi-Directional JEPA is now stable and learning meaningful representations!

 This proves the encoder didn't just learn a trivial identity mapping; it learned a representation of the sine wave's phase and frequency robust enough to predict unseen future data points.

## next steps:

Option A: Increase Complexity (Chaotic Data). We could try this on the Lorenz Attractor or a noisy financial-style dataset. Sine waves are predictable; chaotic systems are much harder for JEPAs.

Option B: Visual Data (Images). We could switch to a small image dataset (like MNIST or CIFAR-10) and implement the spatial masking (patch prediction) used in the original I-JEPA paper.

Option C: Deep Dive into Latents. We haven't really used the "latent variable" z (noise) yet. We could explicitly inject noise to see if the model learns to output a distribution of possible futures (stochastic prediction).

# Lorenz Attractor: chaotic system.

```
The Experiment Setup

    Data: We will generate 3D trajectories (x,y,z) using the Lorenz differential equations.

    Task:

        Input (Context): A sequence of 20 steps.

        Target: The next 20 steps.

    Evaluation: Can the linear probe predict the future state of a chaotic system from the learned embedding?
```
slightly adjusted the BiJEPA hyperparameters (larger hidden size) to handle the increased complexity.

## shared Lorenz data generation.
"""

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to: {seed}")

set_seed(42)

def generate_lorenz_data(batch_size, seq_len=40, dt=0.01, sigma=10, rho=28, beta=8/3):
    """Generates Lorenz Attractor trajectories."""
    data = []
    for _ in range(batch_size):
        x, y, z = np.random.uniform(-15, 15, 3)
        trajectory = []
        for _ in range(seq_len):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx; y += dy; z += dz
            trajectory.append([x, y, z])
        data.append(trajectory)

    # Normalize to roughly [-1, 1] for stability
    data = np.array(data)
    data = (data - np.mean(data)) / np.std(data)
    return torch.tensor(data, dtype=torch.float32)

print("Generating Lorenz Data...")
demo_data = generate_lorenz_data(batch_size=3, seq_len=40)
train_data_pool = generate_lorenz_data(batch_size=2000, seq_len=40)
probe_data = generate_lorenz_data(batch_size=1000, seq_len=40)
test_data = generate_lorenz_data(batch_size=20, seq_len=40)

torch.save({
    'demo_data': demo_data,
    'train_data_pool': train_data_pool,
    'probe_data': probe_data,
    'test_data': test_data
}, 'lorenz_data.pt')
print("Saved to lorenz_data.pt")

# Visualize the Attractor
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    traj = demo_data[i].numpy()
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label=f'Trajectory {i}')
ax.set_title("Lorenz Attractor Samples (Shared Data)")
ax.legend()
plt.show()

"""## classic JEPA (expressive, soft constraints)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import os

# =============================================================================
# 0. Load Shared Data
# =============================================================================
if not os.path.exists('lorenz_data.pt'):
    raise FileNotFoundError("Run the data generation script first!")

shared_data = torch.load('lorenz_data.pt')
train_data_pool = shared_data['train_data_pool']
probe_data = shared_data['probe_data']
test_data = shared_data['test_data']

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# =============================================================================
# 2. Model (Classic JEPA - Expressive Config)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)
        out = self.net(x_flat)
        return out

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class ClassicJEPA(nn.Module):
    def __init__(self, input_len, input_channels=3, embed_dim=32):
        super().__init__()
        flat_input_dim = input_len * input_channels

        # Online Encoder (Context)
        self.online_encoder = Encoder(flat_input_dim, 128, embed_dim)

        # Predictor (Forward Only)
        self.predictor = Predictor(embed_dim, 128)

        # Target Encoder (Target) - EMA Updated
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, momentum=0.995):
        with torch.no_grad():
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x_raw, y_raw):
        # 1. Encode Target (Stop Gradient)
        with torch.no_grad():
            target_embed_y = self.target_encoder(y_raw)

        # 2. Encode Context (Online)
        online_embed_x = self.online_encoder(x_raw)

        # 3. Predict Future (Forward Only)
        pred_y = self.predictor(online_embed_x)

        # 4. Loss (Uni-directional)
        loss = nn.functional.mse_loss(pred_y, target_embed_y)

        return loss

# =============================================================================
# 3. Execution Pipeline
# =============================================================================
seq_len = 40
split_point = 20
embed_dim = 32

set_seed(42)

jepa_model = ClassicJEPA(input_len=split_point, input_channels=3, embed_dim=embed_dim)
optimizer = optim.AdamW(jepa_model.parameters(), lr=5e-4, weight_decay=1e-4)

losses = []

# --- Phase 1: Pre-training ---
print("Phase 1: Pre-training on Lorenz (Classic JEPA)...")
n_steps = 3000
batch_size = 64

for step in range(n_steps):
    indices = torch.randint(0, len(train_data_pool), (batch_size,))
    batch = train_data_pool[indices]

    x_raw = batch[:, :split_point, :]
    y_raw = batch[:, split_point:, :]

    optimizer.zero_grad()
    loss = jepa_model(x_raw, y_raw)
    loss.backward()
    optimizer.step()
    jepa_model.update_target_encoder()

    losses.append(loss.item())

    if step % 500 == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

# Freeze
jepa_model.eval()
for p in jepa_model.parameters(): p.requires_grad = False

# --- Phase 2: Evaluation (Probes) ---
print("\nPhase 2: Evaluation (Probes)...")
train_x = probe_data[:, :split_point, :]
train_y_target = probe_data[:, split_point, :]

# Protocol A
probe_A = nn.Linear(embed_dim, 3)
opt_A = optim.Adam(probe_A.parameters(), lr=0.01)

for epoch in range(200):
    with torch.no_grad(): s_x = jepa_model.online_encoder(train_x)
    preds = probe_A(s_x)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_A.zero_grad(); loss.backward(); opt_A.step()
print(f"Protocol A (Encoder) MSE: {loss.item():.5f}")

# Protocol B
probe_B = nn.Linear(embed_dim, 3)
opt_B = optim.Adam(probe_B.parameters(), lr=0.01)

for epoch in range(200):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
        s_y_hat = jepa_model.predictor(s_x) # Use predictor
    preds = probe_B(s_y_hat)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_B.zero_grad(); loss.backward(); opt_B.step()
print(f"Protocol B (Predictor) MSE: {loss.item():.5f}")

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Loss
plt.subplot(1, 3, 1)
plt.plot(losses, label='Loss', color='tab:blue')
plt.title("Lorenz Training Stability (Classic JEPA)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Forecast X-Coordinate
test_x = test_data[:, :split_point, :]
test_y_true = test_data[:, split_point, :]

with torch.no_grad():
    s_x = jepa_model.online_encoder(test_x)
    s_y_hat = jepa_model.predictor(s_x)
    pred_A = probe_A(s_x)
    pred_B = probe_B(s_y_hat)

plt.subplot(1, 3, 2)
plt.plot(test_y_true[:, 0].numpy(), 'ko-', label='True X', alpha=0.5)
plt.plot(pred_A[:, 0].numpy(), 'rx', label='Proto A', markeredgewidth=2)
plt.plot(pred_B[:, 0].numpy(), 'g^', label='Proto B', markeredgewidth=2)
plt.title("Forecast Accuracy (Classic JEPA)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: 3D Trajectory Reconstruction (Single Sample)
ax = plt.subplot(1, 3, 3, projection='3d')
idx = 0
ctx = test_x[idx].numpy()
true_fut = test_y_true[idx].numpy()
pred_fut = pred_B[idx].numpy()

ax.plot(ctx[:,0], ctx[:,1], ctx[:,2], 'b-', label='Context')
ax.scatter(ctx[-1,0], ctx[-1,1], ctx[-1,2], c='b', s=20)
ax.scatter(true_fut[0], true_fut[1], true_fut[2], c='k', s=50, label='True')
ax.scatter(pred_fut[0], pred_fut[1], pred_fut[2], c='g', marker='^', s=50, label='Pred')
ax.set_title(f"3D State Prediction (Classic)")
ax.legend()

plt.tight_layout()
plt.show()

"""## Expressive (soft constraints) BiJEPA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import os

# =============================================================================
# 0. Load Shared Data
# =============================================================================
if not os.path.exists('lorenz_data.pt'):
    raise FileNotFoundError("Run the data generation script first!")

shared_data = torch.load('lorenz_data.pt')
train_data_pool = shared_data['train_data_pool']
probe_data = shared_data['probe_data']
test_data = shared_data['test_data']

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# =============================================================================
# 2. Model (Expressive: LayerNorm + Decay, No Sphere)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # Flatten: [Batch, Seq, 3] -> [Batch, Seq*3]
        x_flat = x.reshape(x.size(0), -1)
        out = self.net(x_flat)
        # NO L2 Norm (Expressive)
        return out

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class BiJEPA(nn.Module):
    def __init__(self, input_len, input_channels=3, embed_dim=32):
        super().__init__()
        flat_input_dim = input_len * input_channels

        # Increased hidden_dim to 128 for chaotic complexity
        self.online_encoder = Encoder(flat_input_dim, 128, embed_dim)
        self.fwd_predictor = Predictor(embed_dim, 128)
        self.bwd_predictor = Predictor(embed_dim, 128)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, momentum=0.995):
        with torch.no_grad():
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x_raw, y_raw):
        with torch.no_grad():
            target_embed_x = self.target_encoder(x_raw)
            target_embed_y = self.target_encoder(y_raw)

        online_embed_x = self.online_encoder(x_raw)
        online_embed_y = self.online_encoder(y_raw)

        pred_y = self.fwd_predictor(online_embed_x)
        pred_x = self.bwd_predictor(online_embed_y)

        loss_fwd = nn.functional.mse_loss(pred_y, target_embed_y)
        loss_bwd = nn.functional.mse_loss(pred_x, target_embed_x)
        return loss_fwd, loss_bwd

# =============================================================================
# 3. Execution Pipeline
# =============================================================================
seq_len = 40
split_point = 20
embed_dim = 32

set_seed(42)

jepa_model = BiJEPA(input_len=split_point, input_channels=3, embed_dim=embed_dim)
optimizer = optim.AdamW(jepa_model.parameters(), lr=5e-4, weight_decay=1e-4)

losses = []

# --- Phase 1: Pre-training ---
print("Phase 1: Pre-training on Lorenz (Expressive BiJEPA)...")
n_steps = 3000
batch_size = 64

for step in range(n_steps):
    # Random sampling from the pool
    indices = torch.randint(0, len(train_data_pool), (batch_size,))
    batch = train_data_pool[indices]

    x_raw = batch[:, :split_point, :]
    y_raw = batch[:, split_point:, :]

    optimizer.zero_grad()
    loss_fwd, loss_bwd = jepa_model(x_raw, y_raw)
    (loss_fwd + loss_bwd).backward()
    optimizer.step()
    jepa_model.update_target_encoder()

    losses.append((loss_fwd.item(), loss_bwd.item()))

    if step % 500 == 0:
        print(f"Step {step}: Loss Fwd {loss_fwd.item():.4f}")

# Freeze
jepa_model.eval()
for p in jepa_model.parameters(): p.requires_grad = False

# --- Phase 2: Evaluation (Predicting t=21, x,y,z) ---
print("\nPhase 2: Evaluation (Probes)...")
train_x = probe_data[:, :split_point, :]
train_y_target = probe_data[:, split_point, :]

# Protocol A (Encoder Probe)
probe_A = nn.Linear(embed_dim, 3) # Output is 3D (x,y,z)
opt_A = optim.Adam(probe_A.parameters(), lr=0.01)

for epoch in range(200):
    with torch.no_grad(): s_x = jepa_model.online_encoder(train_x)
    preds = probe_A(s_x)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_A.zero_grad(); loss.backward(); opt_A.step()
print(f"Protocol A (Encoder) MSE: {loss.item():.5f}")

# Protocol B (Predictor Probe)
probe_B = nn.Linear(embed_dim, 3)
opt_B = optim.Adam(probe_B.parameters(), lr=0.01)

for epoch in range(200):
    with torch.no_grad():
        s_x = jepa_model.online_encoder(train_x)
        s_y_hat = jepa_model.fwd_predictor(s_x)
    preds = probe_B(s_y_hat)
    loss = nn.functional.mse_loss(preds, train_y_target)
    opt_B.zero_grad(); loss.backward(); opt_B.step()
print(f"Protocol B (Predictor) MSE: {loss.item():.5f}")

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Loss
plt.subplot(1, 3, 1)
f_loss = [l[0] for l in losses]
b_loss = [l[1] for l in losses]
plt.plot(f_loss, label='Fwd', alpha=0.7)
plt.plot(b_loss, label='Bwd', alpha=0.7)
plt.title("Lorenz Training Stability (BiJEPA)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Forecast X-Coordinate
test_x = test_data[:, :split_point, :]
test_y_true = test_data[:, split_point, :]

with torch.no_grad():
    s_x = jepa_model.online_encoder(test_x)
    s_y_hat = jepa_model.fwd_predictor(s_x)
    pred_A = probe_A(s_x)
    pred_B = probe_B(s_y_hat)

plt.subplot(1, 3, 2)
# Plotting just the X-dimension (index 0)
plt.plot(test_y_true[:, 0].numpy(), 'ko-', label='True X (t=21)', alpha=0.5)
plt.plot(pred_A[:, 0].numpy(), 'rx', label='Proto A', markeredgewidth=2)
plt.plot(pred_B[:, 0].numpy(), 'g^', label='Proto B', markeredgewidth=2)
plt.title("Forecast Accuracy (X-Dimension)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: 3D Trajectory Reconstruction (Single Sample)
ax = plt.subplot(1, 3, 3, projection='3d')
idx = 0
ctx = test_x[idx].numpy() # [20, 3]
true_fut = test_y_true[idx].numpy() # [3]
pred_fut = pred_B[idx].numpy() # [3]

# Plot context trajectory
ax.plot(ctx[:,0], ctx[:,1], ctx[:,2], 'b-', label='Context')
ax.scatter(ctx[-1,0], ctx[-1,1], ctx[-1,2], c='b', s=20) # End of context

# Plot True Future
ax.scatter(true_fut[0], true_fut[1], true_fut[2], c='k', s=50, label='True Fut')
# Plot Predicted Future
ax.scatter(pred_fut[0], pred_fut[1], pred_fut[2], c='g', marker='^', s=50, label='Pred Fut')

ax.set_title(f"3D State Prediction (Sample {idx})")
ax.legend()

plt.tight_layout()
plt.show()

"""```
The model became too good at the self-supervised task. It found a way to map the "Past" and "Future" to the same normalized vector perfectly in the abstract space. However, that abstract vector lost some of the fine-grained metric details (like exact amplitude) needed for perfect coordinate regression. This is a common trade-off in SSL: you gain semantic robustness (capturing the "shape" of the chaos) but lose some pixel-perfect precision.
```

# MNIST.

## Now we move from Temporal (Sequence) to Spatial (Image) data.

````
To adapt Bi-Directional JEPA for images, we need to redefine "Past" and "Future" in spatial terms.

    Context (X): The Left Half of the digit.

    Target (Y): The Right Half of the digit.

The Goal: The model must look at the left half of a "3" and predict the embedding of the right half (which completes the curves), and vice-versa.
Changes to the Architecture

    Encoder: Replaced the MLP with a small ConvNet. It must compress a 14x28 image slice into a vector.

    Task: Instead of predicting the next coordinate, the evaluation (Probe) will try to classify the digit (0-9) using only the embedding of the Left Half.

        If the JEPA works, the "Left Half" embedding should contain enough hallucinated information about the "Right Half" to identify the digit.

## shared MINIST dataset.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

# =============================================================================
# 0. Global Reproducibility Configuration
# =============================================================================
FIXED_SEED = 42
FIXED_VISUALIZATION_INDICES = [0, 1, 2, 3, 4] # Always visualize these 5 samples from the test batch

def set_seed(seed=FIXED_SEED):
    """Ensures identical weight initialization and data shuffling."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def get_mnist_loaders(batch_size=256):
    # Reset seed inside loader to ensure shuffling is identical every time this is called
    set_seed(FIXED_SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Train loader shuffled, Test loader fixed
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

print("Configuration loaded. VISUALIZATION_INDICES fixed to:", FIXED_VISUALIZATION_INDICES)

"""## Classic JEPA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import numpy as np

#

# =============================================================================
# 1. Model Components (Classic JEPA)
# =============================================================================
class ConvEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Input: 1x28x14 (Left Half)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 4, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x): return self.net(x)

class GenerativeDecoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Output: 14x28 pixels = 392
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1 * 28 * 14), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).view(-1, 1, 28, 14)

class ClassicJEPA_MNIST(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.online_encoder = ConvEncoder(embed_dim)
        self.predictor = Predictor(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False

    def update_target_encoder(self, momentum=0.99):
        with torch.no_grad():
            for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                t.data = momentum * t.data + (1 - momentum) * o.data

    def forward(self, left_img, right_img):
        # Target: Encode Right Half
        with torch.no_grad(): target_right = self.target_encoder(right_img)
        # Online: Encode Left Half
        online_left = self.online_encoder(left_img)
        # Predict: Left -> Right
        pred_right = self.predictor(online_left)
        # Loss
        return F.mse_loss(pred_right, target_right)

# =============================================================================
# 2. Execution (Classic JEPA)
# =============================================================================
# STRICT SEEDING
set_seed(FIXED_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_mnist_loaders(batch_size=256)

model = ClassicJEPA_MNIST(embed_dim=64).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"Phase 1: Pre-training Classic JEPA (Left -> Right)...")
model.train()
for epoch in range(10):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        left, right = imgs[:, :, :, :14], imgs[:, :, :, 14:]
        optimizer.zero_grad()
        loss = model(left, right)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.5f}")

# --- Phase 2: Linear Probe ---
print("\nPhase 2: Linear Probe Evaluation...")
model.eval()
probe = nn.Linear(64, 10).to(device)
probe_opt = optim.Adam(probe.parameters(), lr=1e-2)
crit = nn.CrossEntropyLoss()

for epoch in range(10):
    correct = 0; total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad(): feats = model.online_encoder(imgs[:, :, :, :14])
        out = probe(feats)
        loss = crit(out, labels)
        probe_opt.zero_grad(); loss.backward(); probe_opt.step()
        _, pred = torch.max(out, 1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
    print(f"Probe Epoch {epoch+1} Acc: {100*correct/total:.2f}%")

# Test Accuracy
correct = 0; total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feats = model.online_encoder(imgs[:, :, :, :14])
        _, pred = torch.max(probe(feats), 1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
print(f"Final Classic JEPA Test Accuracy: {100*correct/total:.2f}%")

# --- Phase 3: Generative Decoder ---
print("\nPhase 3: Training Generative Decoder (Hallucination)...")
# Reseed decoder init for fairness
set_seed(FIXED_SEED)
decoder = GenerativeDecoder(embed_dim=64).to(device)
dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)
mse_crit = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) # Normalize 0-1
        left, right = imgs[:, :, :, :14], imgs[:, :, :, 14:]

        with torch.no_grad(): emb = model.online_encoder(left)
        preds = decoder(emb)
        loss = mse_crit(preds, right)
        dec_opt.zero_grad(); loss.backward(); dec_opt.step()
        total_loss += loss.item()
    print(f"Decoder Epoch {epoch+1}: MSE {total_loss/len(train_loader):.4f}")

# Visualization (FIXED INDICES)
print("Visualizing Classic JEPA Hallucinations...")
model.eval(); decoder.eval()

# Get the exact same test batch
imgs, _ = next(iter(test_loader))
imgs = imgs.to(device)
imgs_norm = (imgs - imgs.min()) / (imgs.max() - imgs.min())
left, right = imgs_norm[:, :, :, :14], imgs_norm[:, :, :, 14:]

with torch.no_grad():
    emb = model.online_encoder(left)
    gen_right = decoder(emb)

# USE FIXED INDICES
indices = FIXED_VISUALIZATION_INDICES
num_samples = len(indices)

plt.figure(figsize=(8, num_samples * 2))
for i, idx in enumerate(indices):
    plt.subplot(num_samples, 3, i*3 + 1); plt.imshow(left[idx, 0].cpu(), cmap='gray'); plt.axis('off')
    if i==0: plt.title("Input (Left)")
    plt.subplot(num_samples, 3, i*3 + 2); plt.imshow(right[idx, 0].cpu(), cmap='gray'); plt.axis('off')
    if i==0: plt.title("True Target")
    plt.subplot(num_samples, 3, i*3 + 3); plt.imshow(gen_right[idx, 0].cpu(), cmap='gray'); plt.axis('off')
    if i==0: plt.title("Classic JEPA Gen")
plt.tight_layout(); plt.show()

"""## Expressive BiJEPA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import numpy as np

# =============================================================================
# 1. Model Components (BiJEPA)
# =============================================================================
# Architecture classes are reused or identical to preserve fairness
#

class ConvEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 4, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x): return self.net(x)

class GenerativeDecoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1 * 28 * 14), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).view(-1, 1, 28, 14)

class BiJEPA_MNIST(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.online_encoder = ConvEncoder(embed_dim)
        self.fwd_predictor = Predictor(embed_dim)
        self.bwd_predictor = Predictor(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False

    def update_target_encoder(self, momentum=0.99):
        with torch.no_grad():
            for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                t.data = momentum * t.data + (1 - momentum) * o.data

    def forward(self, left_img, right_img):
        with torch.no_grad():
            t_left = self.target_encoder(left_img)
            t_right = self.target_encoder(right_img)
        o_left = self.online_encoder(left_img)
        o_right = self.online_encoder(right_img)
        loss_fwd = F.mse_loss(self.fwd_predictor(o_left), t_right)
        loss_bwd = F.mse_loss(self.bwd_predictor(o_right), t_left)
        return loss_fwd, loss_bwd

# =============================================================================
# 2. Execution (BiJEPA)
# =============================================================================
# STRICT SEEDING
set_seed(FIXED_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_mnist_loaders(batch_size=256)

model = BiJEPA_MNIST(embed_dim=64).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"\nPhase 1: Pre-training BiJEPA (Left <-> Right)...")
model.train()
for epoch in range(10):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        left, right = imgs[:, :, :, :14], imgs[:, :, :, 14:]
        optimizer.zero_grad()
        loss_fwd, loss_bwd = model(left, right)
        loss = loss_fwd + loss_bwd
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.5f}")

# --- Phase 2: Linear Probe ---
print("\nPhase 2: Linear Probe Evaluation...")
model.eval()
probe = nn.Linear(64, 10).to(device)
probe_opt = optim.Adam(probe.parameters(), lr=1e-2)
crit = nn.CrossEntropyLoss()

for epoch in range(10):
    correct = 0; total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad(): feats = model.online_encoder(imgs[:, :, :, :14])
        out = probe(feats)
        loss = crit(out, labels)
        probe_opt.zero_grad(); loss.backward(); probe_opt.step()
        _, pred = torch.max(out, 1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
    print(f"Probe Epoch {epoch+1} Acc: {100*correct/total:.2f}%")

# Test Accuracy
correct = 0; total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feats = model.online_encoder(imgs[:, :, :, :14])
        _, pred = torch.max(probe(feats), 1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
print(f"Final BiJEPA Test Accuracy: {100*correct/total:.2f}%")

# --- Phase 3: Generative Decoder ---
print("\nPhase 3: Training Generative Decoder (Hallucination)...")
# Reseed decoder init for fairness
set_seed(FIXED_SEED)
decoder = GenerativeDecoder(embed_dim=64).to(device)
dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)
mse_crit = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
        left, right = imgs[:, :, :, :14], imgs[:, :, :, 14:]

        with torch.no_grad(): emb = model.online_encoder(left)
        preds = decoder(emb)
        loss = mse_crit(preds, right)
        dec_opt.zero_grad(); loss.backward(); dec_opt.step()
        total_loss += loss.item()
    print(f"Decoder Epoch {epoch+1}: MSE {total_loss/len(train_loader):.4f}")

# Visualization (FIXED INDICES)
print("Visualizing BiJEPA Hallucinations...")
model.eval(); decoder.eval()

# Get the exact same test batch
imgs, _ = next(iter(test_loader))
imgs = imgs.to(device)
imgs_norm = (imgs - imgs.min()) / (imgs.max() - imgs.min())
left, right = imgs_norm[:, :, :, :14], imgs_norm[:, :, :, 14:]

with torch.no_grad():
    emb = model.online_encoder(left)
    gen_right = decoder(emb)

# USE FIXED INDICES
indices = FIXED_VISUALIZATION_INDICES
num_samples = len(indices)

plt.figure(figsize=(8, num_samples * 2))
for i, idx in enumerate(indices):
    plt.subplot(num_samples, 3, i*3 + 1); plt.imshow(left[idx, 0].cpu(), cmap='gray'); plt.axis('off')
    if i==0: plt.title("Input (Left)")
    plt.subplot(num_samples, 3, i*3 + 2); plt.imshow(right[idx, 0].cpu(), cmap='gray'); plt.axis('off')
    if i==0: plt.title("True Target")
    plt.subplot(num_samples, 3, i*3 + 3); plt.imshow(gen_right[idx, 0].cpu(), cmap='gray'); plt.axis('off')
    if i==0: plt.title("BiJEPA Gen")
plt.tight_layout(); plt.show()

"""```
Random Guessing: ~10% accuracy.

Pixel Statistics: If you just trained a logistic regression on the raw pixels of the left half, you would likely get around 50-60%.

Our Result (73%): This proves the Encoder didn't just memorize "ink density." It learned Semantic Identity. To predict the right half of the image effectively, the model had to implicitly figure out: "This curve looks like the start of a 3, so the other side must close the loop."

```
Visualizing the "Hallucination"

Next, we will freeze the trained JEPA encoder and train a small network to take the Left-Half Embedding and generate the pixels of the Right Half.

When looking at the generated images, look for these specific behaviors:

    The "Average" Blur: If the model is uncertain (e.g., input is a vertical bar |), it might output a blurry cloud. This is actually a good sign—it means the model knows multiple futures are possible and is averaging them (e.g., it could be a 0, a 1, or a 6).

    Structural Completion:

        Loop Closing: If the left half is (, does the model generate ) to make a 0?

        Diagonal Strokes: If the left half is the top of a 7, does the right half continue the diagonal line downward?

    Identity Preservation: Does the generated right half actually look like it belongs to the same digit class as the left half?
```

# Noise injection (future work)

## Deep Dive into Latents. We haven't really used the "latent variable" z (noise) yet. We could explicitly inject noise to see if the model learns to output a distribution of possible futures (stochastic prediction).
"""

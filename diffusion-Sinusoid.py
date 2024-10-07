# -*- coding: utf-8 -*-
"""diffusion_Example.ipynb

Original file is located at
https://colab.research.google.com/drive/15OUuRbUlxe-Qb2kMrNMGU2zGindRgHox

**Author:** Dr. Shahriar Hossain <br>
**YT Channel:** https://www.youtube.com/@C4A <br>
**Web:** https://computing4all.com/ <br>
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Defining the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
else:
    print("No NVIDIA driver found. Using CPU")

# Original sinusoidal data
n_features = 100
n_samples = 100
x = np.linspace(0, 6 * np.pi, n_features)  # 3 complete cycles
original = torch.tensor([np.sin(x + 2 * np.pi * i / n_samples) for i in range(n_samples)], dtype=torch.float32)

# Diffusion noise factor
noise_factor = 0.005
# Number of diffusion steps
n_steps = 50

data = original
diffused_data_steps = [data]

# Adding noise through diffusion steps
for _ in range(n_steps):
    data = data + noise_factor * torch.randn_like(data)
    diffused_data_steps.append(data)

# Convert the diffused data steps into a dataset
# Each sample is a pair of subsequent diffused data steps
diffused_data_pairs = [torch.stack((diffused_data_steps[i+1], diffused_data_steps[i]), dim=1)
                       for i in range(n_steps)]
diffused_data = torch.cat(diffused_data_pairs, dim=0)
dataset = TensorDataset(*torch.unbind(diffused_data, dim=1))

# Create a DataLoader to handle batching of the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Neural Network
model = nn.Sequential(
    nn.Linear(n_features, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, n_features)
)

# Move the model to device
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()

# Training loop
for i in range(1, 50):
    model.train()  # ensure the model is in training mode

    for batch_noisy, batch_data in dataloader:
        # Move data to the device
        batch_noisy, batch_data = batch_noisy.to(device), batch_data.to(device)

        optimizer.zero_grad()
        batch_denoised = model(batch_noisy)
        loss = loss_func(batch_denoised, batch_data)
        loss.backward()
        optimizer.step()

    if i % 5 == 0:
        print(f'Epoch {i}, Loss: {loss.item()}')

# Data generation phase
# Start with random noise instead of a sinusoid
generated_data = torch.randn((1, n_features), dtype=torch.float32).to(device)

# Move initial generated data to CPU for visualization
generated_data_cpu = generated_data.detach().cpu().numpy().flatten()

# Plot initial data
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(range(n_features), generated_data_cpu, lw=2)

ax.set_xlim(0, n_features - 1)
ax.set_ylim(-2, 2)
ax.set_title('Generated Sinusoid Over Diffusion Steps - Step 0')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Value')

# Pause to show the initial state before animation starts
plt.pause(1)

def init():
    # Set the initial line to the generated random data (optional, as we have already plotted it)
    line.set_data(range(n_features), generated_data_cpu)
    return line,

def update(step):
    global generated_data
    # Add noise and use model to denoise
    noised_data = generated_data + noise_factor * torch.randn_like(generated_data)
    generated_data = model(noised_data)

    # Move to CPU for visualization
    generated_data_cpu = generated_data.detach().cpu().numpy().flatten()
    
    # Update the line and title
    line.set_data(range(n_features), generated_data_cpu)
    ax.set_title(f'Generated Sinusoid Over Diffusion Steps - Step {step + 1}')
    return line,

ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=False, interval=500, repeat=False)

plt.show()

# Display final generated sinusoidal data
generated_data_cpu = generated_data.detach().cpu().numpy().flatten()
print("Final generated sinusoid:", generated_data_cpu)


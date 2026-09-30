# data/generate_data.py
import os
import torch
import numpy as np

from utils import load_config
from sim import simulate_jaynes_cummings
from train import parse_args

def generate_multi_trajectory_dataset():

    args = parse_args()

    num_trajectories = args.num_trajectories
    noise_level = args.noise_level
    save_path = args.save_path

    print("Loading physical constants from config.yaml...")
    config = load_config()
    phys = config['physics']
    
    wc = phys['cavity_freq']
    wa = phys['atom_freq']
    t_max = phys['t_max']
    
    all_inputs = []  # Will store [t, g, kappa, gamma]
    all_targets = [] # Will store corresponding P_e
    
    print(f"Generating {num_trajectories} distinct quantum trajectories (Noise level: {noise_level})...")
    
    for i in range(num_trajectories):
        # Sample random parameters within realistic physical bounds
        g_val = np.random.uniform(0.1, 0.5)
        kappa_val = np.random.uniform(0.01, 0.1)
        gamma_val = np.random.uniform(0.01, 0.1)
        
        # Run QuTiP engine for this specific parameter set
        tlist, p_exact, _, _ = simulate_jaynes_cummings(
            wc=wc, wa=wa, 
            g=g_val, kappa=kappa_val, gamma=gamma_val, 
            t_max=t_max
        )
        
        # Add slight Gaussian noise
        noise = np.random.normal(0, noise_level, size=len(p_exact))
        p_noisy = np.clip(p_exact + noise, -0.1, 1.1) # Realistic probability bounds clipping
        
        # Create input features for every time step: [t, g, kappa, gamma]
        for t, p in zip(tlist, p_noisy):
            all_inputs.append([t, g_val, kappa_val, gamma_val])
            all_targets.append([p])
            
    # Convert to large PyTorch tensors
    inputs_tensor = torch.tensor(all_inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)
    
    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'inputs': inputs_tensor, 'targets': targets_tensor}, save_path)
    
    print(f"Successfully generated dataset with {len(inputs_tensor)} total data points saved to {save_path}")

if __name__ == "__main__":
    
    np.random.seed(42)
    generate_multi_trajectory_dataset()
# /sim/generate_data.py
import os
import yaml
import torch
import numpy as np
from sim import simulate_jaynes_cummings
from utils import load_config

def generate_quantum_data():
    config = load_config()  # Now called from utils
    phys = config['physics']
    ml = config['ml']
    
    print("Running physics engine...")
    tlist, p_excited = simulate_jaynes_cummings(
        wc=phys['cavity_freq'], wa=phys['atom_freq'], 
        g=phys['coupling_g'], kappa=phys['kappa'], 
        gamma=phys['gamma'], 
        t_max=phys['t_max'], t_steps=phys['t_steps']
    )
    
    # Inject Measurement Noise
    np.random.seed(42) 
    noisy_data = p_excited + np.random.normal(0, ml['noise_std'], size=len(tlist))
    
    # Export for PyTorch
    os.makedirs("data", exist_ok=True)
    save_path = "data/simulated_trajectory.pt"
    
    torch.save({
        't': torch.tensor(tlist, dtype=torch.float32),
        'y_true': torch.tensor(p_excited, dtype=torch.float32),
        'y_noisy': torch.tensor(noisy_data, dtype=torch.float32)
    }, save_path)
    
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    generate_quantum_data()
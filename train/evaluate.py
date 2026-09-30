# train/evaluate.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import ParameterConditionedSurrogate
from sim.engine import simulate_jaynes_cummings
from utils.config import load_config

def evaluate_surrogate(model_path="checkpoints/surrogate_final.pth"):
    device = torch.device("cpu") # CPU is great for fast inference checks
    print(f"Loading surrogate model from {model_path}...")
    
    # Initialize and load weights
    model = ParameterConditionedSurrogate().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    config = load_config()
    phys = config['physics']
    wc, wa, t_max = phys['cavity_freq'], phys['atom_freq'], phys['t_max']
    
    # Define test parameter sets (chosen to test distinct physical regimes)
    test_cases = [
        {"g": 0.35, "kappa": 0.05, "gamma": 0.02, "label": "Balanced Regime"},
        {"g": 0.15, "kappa": 0.08, "gamma": 0.08, "label": "High Dissipation Regime"},
        {"g": 0.48, "kappa": 0.01, "gamma": 0.01, "label": "Strong Coupling / Low Loss"}
    ]
    
    os.makedirs("evaluation_plots", exist_ok=True)
    
    with torch.no_grad():
        for i, case in enumerate(test_cases):
            g, kappa, gamma = case["g"], case["kappa"], case["gamma"]
            print(f"Evaluating Case {i+1} ({case['label']}): g={g}, kappa={kappa}, gamma={gamma}")
            
            # 1. Generate ground truth using QuTiP engine
            tlist, p_exact, _, _ = simulate_jaynes_cummings(
                wc=wc, wa=wa, g=g, kappa=kappa, gamma=gamma, t_max=t_max
            )
            
            # 2. Prepare inputs for the surrogate model
            # tlist shape: (N,), we need shape (N, 1)
            t_tensor = torch.tensor(tlist, dtype=torch.float32).unsqueeze(1).to(device)
            # Parameters need to match shape (N, 3): [g, kappa, gamma] repeated for every time step
            params_vals = torch.tensor([[g, kappa, gamma]], dtype=torch.float32).repeat(len(tlist), 1).to(device)
            
            # 3. Predict using surrogate model
            p_pred_tensor = model(t_tensor, params_vals)
            p_pred = p_pred_tensor.squeeze().cpu().numpy()
            
            # 4. Compute error metrics
            mse = np.mean((p_exact - p_pred) ** 2)
            print(f"  -> Test MSE: {mse:.2e}")
            
            # 5. Plot comparison
            plt.figure(figsize=(8, 4))
            plt.plot(tlist, p_exact, label="QuTiP (Ground Truth)", color="black", linewidth=2)
            plt.scatter(tlist, p_pred, label="Surrogate Model", color="crimson", marker = ".")
            plt.title(f"{case['label']} (MSE: {mse:.2e})")
            plt.xlabel("Time ($t$)")
            plt.ylabel("Atomic Excitation Probability ($P_e$)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = f"evaluation_plots/test_case_{i+1}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved plot to {plot_path}\n")

if __name__ == "__main__":
    evaluate_surrogate()
import torch
import torch.nn as nn
import torch.optim as optim
from params import JCParams
from model import ParameterConditionedSurrogate
from torch.optim.lr_scheduler import CosineAnnealingLR
from train import parse_args


class SurrogateFitter:
    def __init__(self, model_path="checkpoints/surrogate_final.pth", device="cpu"):
        self.device = torch.device(device)
        print(f"Loading pre-trained surrogate model from {model_path}...")
        
        # 1. Instantiate the surrogate architecture and move to device
        self.model = ParameterConditionedSurrogate().to(self.device)
        
        # 2. Load the trained checkpoint weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 3. Freeze weights since we are optimizing physical parameters, not network weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def fit(self, tlist, params_init, target_curve, lr=None, min_lr=None, steps=None):
        # Initialize parameter container module with your initial guesses
        # (Assuming params_init is a tuple/list like [g, kappa, gamma])

        args = parse_args()

        lr = lr if lr is not None else args.lr_fit
        min_lr = min_lr if min_lr is not None else args.min_fit_lr
        steps = steps if steps is not None else args.steps_fit
        
        params = JCParams(
            params_init=params_init, 
            device=self.device
        )
        
        # Pass the module's parameters to the optimizer
        optimizer = optim.Adam(params.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
        criterion = nn.MSELoss()
        
        t_tensor = torch.tensor(tlist, dtype=torch.float32, device=self.device).unsqueeze(1)
        target_tensor = torch.tensor(target_curve, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        print("Starting parameter optimization...")
        for step in range(steps):
            optimizer.zero_grad()
            
            # Get physically constrained parameters via softplus
            p_dict = params.constrained()
            g, kappa, gamma = p_dict["g"], p_dict["kappa"], p_dict["gamma"]
            
            # Broadcast to match time series shape (N, 3)
            params_batch = torch.cat([g, kappa, gamma]).repeat(len(tlist), 1)
            
            # Predict curve using surrogate
            predicted_curve = self.model(t_tensor, params_batch)
            
            loss = criterion(predicted_curve, target_tensor)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0 or step == 0:
                print(f"Step {step+1}/{steps} | Loss: {loss.item():.5f} | "
                      f"g: {g.item():.4f}, kappa: {kappa.item():.4f}, gamma: {gamma.item():.4f}")
            
        # Return final optimized parameters as standard Python floats
        return {k: v.item() for k, v in params.constrained().items()}
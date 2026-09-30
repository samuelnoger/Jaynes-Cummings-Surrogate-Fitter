import torch
import torch.nn as nn
from typing import Dict

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse of the softplus function to properly initialize constrained parameters."""
    return torch.log(torch.exp(x) - 1.0)

class JCParams(nn.Module):
    def __init__(
        self,
        params_init: Dict[str, float],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        # Convert initial physical guesses to tensors
        g_tensor = torch.tensor([params_init['g']], device=device, dtype=dtype)
        kappa_tensor = torch.tensor([params_init['kappa']], device=device, dtype=dtype)
        gamma_tensor = torch.tensor([params_init['gamma']], device=device, dtype=dtype)

        # Apply inv_softplus so that softplus(raw_param) equals our exact initial guess!
        # We use clamp_min to prevent math errors if a guess is too close to 0.
        self.raw_g = nn.Parameter(inv_softplus(g_tensor.clamp_min(1e-6)))
        self.raw_kappa = nn.Parameter(inv_softplus(kappa_tensor.clamp_min(1e-6)))
        self.raw_gamma = nn.Parameter(inv_softplus(gamma_tensor.clamp_min(1e-6)))

    def constrained(self) -> Dict[str, torch.Tensor]:
        """Maps unconstrained raw parameters back to physical values via softplus."""
        return {
            "g": torch.nn.functional.softplus(self.raw_g),
            "kappa": torch.nn.functional.softplus(self.raw_kappa),
            "gamma": torch.nn.functional.softplus(self.raw_gamma),
        }
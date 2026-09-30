# /model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Frequency bands
        self.register_buffer('freqs', torch.logspace(0, 3, num_frequencies))

    def forward(self, x):
        # x shape: (batch_size, 1)
        # returns shape: (batch_size, 2 * num_frequencies)
        args = x * self.freqs * torch.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim, condition_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # FiLM generator produces scale (gamma) and shift (beta)
        self.gamma_fc = nn.Linear(condition_dim, hidden_dim)
        self.beta_fc = nn.Linear(condition_dim, hidden_dim)

    def forward(self, x, cond):
        # x: hidden features, cond: embedded parameters [g, kappa, gamma]
        h = F.tanh(self.linear(x))
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        # Apply Feature-wise Linear Modulation: h' = gamma * h + beta
        return gamma * h + beta

class ParameterConditionedSurrogate(nn.Module):
    def __init__(self, hidden_neurons=64, num_frequencies=16):
        super().__init__()
        
        # 1. Encoders
        self.time_embed = SinusoidalEmbedding(num_frequencies)
        time_input_dim = 2 * num_frequencies
        
        # Parameters [g, kappa, gamma] (3 parameters)
        self.param_embed = nn.Sequential(
            nn.Linear(3, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons)
        )
        
        # 2. Network Backbone with FiLM conditioning
        self.input_layer = nn.Linear(time_input_dim, hidden_neurons)
        
        self.film1 = FiLMBlock(hidden_neurons, hidden_neurons)
        self.film2 = FiLMBlock(hidden_neurons, hidden_neurons)
        
        # 3. Output layer maps back to P_e probability
        self.output_layer = nn.Linear(hidden_neurons, 1)

    def forward(self, t, params):
        """
        Forward pass maps time 't' and physical parameters [g, kappa, gamma] to predicted P_e.
        t expected shape: (batch_size, 1)
        params expected shape: (batch_size, 3) -> [g, kappa, gamma]
        """
        # Embed time and parameters
        t_enc = self.time_embed(t)
        cond = self.param_embed(params)
        
        # Initial feature pass
        h = F.tanh(self.input_layer(t_enc))
        
        # Pass through FiLM modulation blocks
        h = self.film1(h, cond)
        h = self.film2(h, cond)
        
        # Final prediction
        p_e_pred = self.output_layer(h)
        return p_e_pred
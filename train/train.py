# /train/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ParameterConditionedSurrogate
from train import parse_args

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"--- Parameter-Conditioned Surrogate Training ---")
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading surrogate dataset from {args.data_path}...")
    dataset_file = torch.load(args.data_path)
    inputs = dataset_file['inputs']   
    targets = dataset_file['targets'] 
    
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Model and Optimizer
    model = ParameterConditionedSurrogate(
        hidden_neurons=args.hidden_neurons
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=getattr(args, 'min_lr', 1e-6))
    criterion = nn.MSELoss()

    # 2. Wrap the epochs loop with tqdm
    print(f"Starting surrogate training for {args.epochs} epochs...")
    pbar = tqdm(range(1, args.epochs + 1), desc="Training Surrogate")

    for epoch in pbar:
        model.train()
        running_loss = 0.0
        
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            t = batch_inputs[:, 0:1]         
            params = batch_inputs[:, 1:4]    
            
            predictions = model(t, params)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_inputs.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        current_lr = scheduler.get_last_lr()[0]
        
        # 3. Update the progress bar postfix with live training metrics
        pbar.set_postfix({
            'Loss': f"{epoch_loss:.2e}",
            'LR': f"{current_lr:.2e}"
        })

    # Save final checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, "surrogate_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Surrogate training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
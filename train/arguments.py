# /train/arguments.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Parameter-Conditioned Surrogate Model")
    
    # 1. Hardware / Device settings
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Hardware device for training (cpu, cuda, mps)')
    
    # 2. Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate for the Adam optimizer')
    parser.add_argument('--min-lr', type=float, default=1e-6,help='Minimum learning rate for the scheduler')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training data loader')
    
    # 3. Model Architecture
    parser.add_argument('--hidden-neurons', type=int, default=64, help='Number of neurons per hidden layer')
    
    # 4. I/O Paths
    parser.add_argument('--data-path', type=str, default='data/surrogate_dataset.pt', help='Path to the multi-trajectory surrogate dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='Directory to save trained model weights')

    # Data Generation Parameters (if needed for on-the-fly generation)
    parser.add_argument('--num-trajectories', type=int, default=1000, help='Number of randomized trajectories to simulate')
    parser.add_argument('--noise-level', type=float, default=0.02, help='Standard deviation of Gaussian noise added to trajectories')
    parser.add_argument('--save-path', type=str, default='data/surrogate_dataset.pt', help='Path to save the generated dataset')

    # Fitting arguments (for parameter fitting)
    parser.add_argument('--steps-fit', type=int, default=500, help='Number of optimization steps for fitting parameters')
    parser.add_argument('--lr-fit', type=float, default=0.01, help='Learning rate for fitting parameters')
    parser.add_argument('--min-fit-lr', type=float, default=1e-5, help='Minimum learning rate for fitting scheduler')
    
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    # Test the parser if run directly
    args = parse_args()
    print(args)
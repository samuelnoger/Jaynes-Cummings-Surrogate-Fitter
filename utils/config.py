# /utils/config.py
import os
import yaml

def load_config(config_path="config.yaml"):
    """Loads parameters from the central YAML config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
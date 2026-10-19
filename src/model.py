import os
import yaml
from ultralytics import YOLO

def get_perception_model(config_path="config/config.yaml"):
    """
    Config file se model specifications load karke YOLO instance return karta hai.
    """
    # Config file read karna
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    model_name = config['model_params']['model_type']
    
    print(f"[SYSTEM LOG] Initializing {model_name} architecture...")
    model = YOLO(model_name)
    return model

if __name__ == "__main__":
    # Test script locally
    model = get_perception_model()
    print("[SYSTEM LOG] Model initialized successfully!")
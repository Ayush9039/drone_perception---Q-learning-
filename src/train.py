import os
import argparse
import yaml
from model import get_perception_model

def run_training(config_path="config/config.yaml"):
    # 1. Config parameters load karna
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    model_params = config['model_params']
    
    # 2. Get initialized model
    model = get_perception_model(config_path)
    
    # 3. Path setup for Colab data configuration
    # Note: YOLO ko train karne ke liye ek 'data.yaml' file chahiye hoti hai jo dataset ke paths batati hai
    data_yaml_path = os.path.join(config['colab_paths']['dataset_dir'], "data.yaml")
    
    print("\n--- Starting Model Training ---")
    print(f"Epochs: {model_params['epochs']}")
    print(f"Batch Size: {model_params['batch_size']}")
    print(f"Image Size: {model_params['img_size']}")
    print("--------------------------------\n")
    
    # 4. Actual Training Loop (Colab GPU utilizes here)
    results = model.train(
        data=data_yaml_path,
        epochs=model_params['epochs'],
        batch=model_params['batch_size'],
        imgsz=model_params['img_size'],
        device=0,  # 0 means GPU on Colab
        project=config['colab_paths']['drive_sync_dir'],
        name="drone_perception_run"
    )
    
    print("[SYSTEM LOG] Training completed! Weights saved directly to Google Drive.")

if __name__ == "__main__":
    run_training()
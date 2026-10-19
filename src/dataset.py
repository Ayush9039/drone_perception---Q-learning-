import os
import zipfile

def setup_cloud_dataset(dataset_slug="chintanmistry/drone-obstacle-detection", download_dir="/content/dataset"):
    """
    Google Colab par direct Kaggle se dataset download aur extract karne ka function.
    """
    print(f"[DATA LOG] Checking for dataset source: {dataset_slug}")
    
    # Colab par temporary directory banana
    os.makedirs(download_dir, exist_ok=True)
    
    # Kaggle command run karne ke liye system call (Colab terminal pr chalega)
    # Note: Colab par chalane se pehle kaggle.json verify karna hoga
    try:
        print("[DATA LOG] Downloading dataset from Kaggle...")
        os.system(f"kaggle datasets download -d {dataset_slug} -p {download_dir}")
        
        # Zip file ko unzip karna
        zip_path = os.path.join(download_dir, f"{dataset_slug.split('/')[-1]}.zip")
        if os.path.exists(zip_path):
            print("[DATA LOG] Extracting dataset files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(zip_path) # Extraction ke baad zip file delete karke space save karna
            print(f"[DATA LOG] Dataset ready at: {download_dir}")
        else:
            print("[ERROR] Zip file download fail ho gayi. Kaggle credentials check karein.")
            
    except Exception as e:
        print(f"[ERROR] Dataset setup me problem aayi: {str(e)}")

if __name__ == "__main__":
    print("Dataset module ready. In Colab environment, call 'setup_cloud_dataset()'")
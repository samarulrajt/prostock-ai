#!/usr/bin/env python3
"""Upload ProStock AI model to Hugging Face Hub.

Prerequisites:
1. Create a Hugging Face account at https://huggingface.co/join
2. Create a repository at https://huggingface.co/new-repo with name: samarulraj/indian-stock-lstm
3. Login: huggingface-cli login (or use web interface)

Then run: python3 upload_hf.py
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
HF_REPO_ID = "samarulraj/indian-stock-lstm"  # Your Hugging Face repo ID
LOCAL_DIR = "/Users/samarulraj/works/simple-model"  # Local path with model files


def upload_model():
    """Upload model and scaler to Hugging Face Hub."""
    api = HfApi()
    
    # Try to create repo (will succeed if it exists, or create it)
    try:
        print(f"Creating/verifying repo: {HF_REPO_ID}...")
        create_repo(repo_id=HF_REPO_ID, exist_ok=True)
        print(f"✅ Repo ready: {HF_REPO_ID}")
    except Exception as e:
        print(f"⚠️ Repo setup note: {str(e)[:100]}")
        print("   Make sure you're logged in: huggingface-cli login")
    
    # Upload files
    files_to_upload = [
        "pro_model.h5",      # Trained Keras model
        "pro_scaler.pkl",  # Fitted scaler
    ]
    
    for file in files_to_upload:
        local_path = os.path.join(LOCAL_DIR, file)
        if os.path.exists(local_path):
            print(f"Uploading {file}...")
            try:
                api.upload_file(
                    repo_id=HF_REPO_ID,
                    path_in_repo=file,
                    path_or_fileobj=local_path,
                    commit_message=f"Add {file}",
                )
                print(f"✅ {file} uploaded successfully!")
            except Exception as e:
                print(f"⚠️ Upload note for {file}: {str(e)[:80]}")
        else:
            print(f"❌ {file} not found at {local_path}")
    
    # Upload README if it exists
    readme_path = os.path.join(LOCAL_DIR, "README.md")
    if os.path.exists(readme_path):
        print("Uploading README.md...")
        api.upload_file(
            repo_id=HF_REPO_ID,
            path_in_repo="README.md",
            path_or_fileobj=readme_path,
            commit_message="Add README.md",
        )
        print("✅ README.md uploaded successfully!")
    
    print(f"\n🌐 Repository URL: https://huggingface.co/{HF_REPO_ID}")
    print("🔑 To use the model, users can run:")
    print("  from huggingface_hub import hf_hub_download")
    print("  model_path = hf_hub_download(repo_id=HF_REPO_ID, filename='pro_model.h5')")
    print("\n💡 Prerequisites:")
    print("  1. Create account at https://huggingface.co/join")
    print("  2. Create repo: samarulraj/indian-stock-lstm")
    print("  3. Run: huggingface-cli login (to save token)")


if __name__ == "__main__":
    print(f"Uploading to Hugging Face Hub: {HF_REPO_ID}")
    print(f"Local directory: {LOCAL_DIR}")
    print("-" * 60)
    upload_model()
UPLOAD_EOF
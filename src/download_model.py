from huggingface_hub import snapshot_download
import os


model_name = "microsoft/deberta-v3-small"
local_dir = os.path.join("..", "models", model_name.split('/')[-1])

print(f"--- Downloading model: {model_name} ---")
print(f"Saving to: {local_dir}")


snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False 
)

print(f" Download complete!")
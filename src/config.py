import torch

CONFIG = {
    # Paths (relative to src/)
    "test_csv": "../dataset/test.csv",
    "test_image_dir": "../dataset/test_images",

    # Models
    "text_model": "distilroberta-base",
    "image_model": "efficientnet_b0",

    # Parameters
    "batch_size": 32,
    "img_size": 224,
    "max_text_len": 128,
    "dropout": 0.2,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

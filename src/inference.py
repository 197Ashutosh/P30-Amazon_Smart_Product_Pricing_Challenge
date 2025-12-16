import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm
from config import CONFIG
from dataset import ProductDataset
from model import ProductPricePredictor

def run_inference():
    test_df = pd.read_csv(CONFIG["test_csv"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = ProductDataset(
        test_df,
        CONFIG["test_image_dir"],
        tokenizer,
        transform,
        CONFIG["max_text_len"]
    )

    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = ProductPricePredictor(
        CONFIG["text_model"],
        CONFIG["image_model"],
        CONFIG["dropout"]
    )

    model.load_state_dict(torch.load("../best_model.pth", map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            preds = model(
                batch["input_ids"].to(CONFIG["device"]),
                batch["attention_mask"].to(CONFIG["device"]),
                batch["image"].to(CONFIG["device"]),
                batch["ipq"].to(CONFIG["device"]),
            )
            predictions.extend(np.expm1(preds.cpu().numpy()))

    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": np.maximum(0, predictions)
    })

    submission.to_csv("submission.csv", index=False)
    print("submission.csv generated successfully")

if __name__ == "__main__":
    run_inference()

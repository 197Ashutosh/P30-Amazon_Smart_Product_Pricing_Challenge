import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
import re

def extract_ipq(text):
    text = str(text).lower()
    match = re.search(
        r'(?:pack of|count of|pk of|pack|count|pk)\s*(\d+)|(\d+)\s*(?:count|ct|pk)',
        text
    )
    if match:
        return float(match.group(1) or match.group(2))
    return 1.0

class ProductDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, image_transform, max_len, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            p01, p99 = self.df['price'].quantile([0.01, 0.99])
            self.df['price_clipped'] = self.df['price'].clip(p01, p99)
            self.df['log_price'] = np.log1p(self.df['price_clipped'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['catalog_content'])

        ipq = torch.tensor([extract_ipq(text)], dtype=torch.float32)

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        img_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), "white")

        image = self.image_transform(image)

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image": image,
            "ipq": ipq
        }

        if not self.is_test:
            item["target"] = torch.tensor(row["log_price"], dtype=torch.float32)

        return item

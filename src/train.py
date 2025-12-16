import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm
from config import CONFIG
from dataset import ProductDataset
from model import ProductPricePredictor

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

def run_training():
    set_seed(CONFIG['seed'])
    
    # 1. Load Data and setup splits
    full_train_df = pd.read_csv(CONFIG['train_csv'])
    full_train_df = full_train_df[full_train_df['sample_id'] != 279285].reset_index(drop=True)
    train_df, val_df = train_test_split(full_train_df, test_size=0.1, random_state=CONFIG['seed'])
    
    # 2. Setup tokenizer and transforms
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Create Datasets and DataLoaders
    train_dataset = ProductDataset(train_df, CONFIG["train_image_dir"], tokenizer, train_transform, is_test=False)
    val_dataset = ProductDataset(val_df, CONFIG["train_image_dir"], tokenizer, val_transform, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    
    # 4. Initialize Model, Load Weights, and Setup Fine-Tuning Optimizer
    model = ProductPricePredictor(CONFIG["text_model"], CONFIG["image_model"], CONFIG['dropout'])
    model.load_state_dict(torch.load("../best_model.pth", map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    print("✅ Loaded best_model.pth for fine-tuning!")

    criterion = nn.MSELoss()

    # --- THIS IS THE KEY CHANGE FOR FINE-TUNING ---
    # Create parameter groups for differential learning rates
    optimizer_params = [
        {"params": model.text_model.parameters(), "lr": 2e-6},  # Very low LR for the text backbone
        {"params": model.image_model.parameters(), "lr": 2e-6}, # Very low LR for the image backbone
        {"params": model.regressor_head.parameters(), "lr": 1e-5}  # Slightly higher LR for the custom head
    ]
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=CONFIG["weight_decay"])
    # --- END OF KEY CHANGE ---
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-8)
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. Training loop
    best_smape_score = 50.2258 # Start with your best score to only save better models
    epochs_no_improve = 0
    patience = 3 # Use a little more patience for fine-tuning

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Fine-Tuning]"):
            optimizer.zero_grad()
            
            # Note: We are NOT using brand_id here
            input_ids = batch['input_ids'].to(CONFIG["device"])
            attention_mask = batch['attention_mask'].to(CONFIG["device"])
            image = batch['image'].to(CONFIG["device"])
            ipq = batch['ipq'].to(CONFIG["device"])
            targets = batch['target'].to(CONFIG["device"])

            with torch.amp.autocast(device_type='cuda'):
                predictions = model(input_ids, attention_mask, image, ipq)
                loss = criterion(predictions, targets)
            
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        # 6. Validation and SMAPE calculation
        model.eval()
        val_loss = 0; all_targets_log, all_preds_log = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
                input_ids = batch['input_ids'].to(CONFIG["device"])
                attention_mask = batch['attention_mask'].to(CONFIG["device"])
                image = batch['image'].to(CONFIG["device"])
                ipq = batch['ipq'].to(CONFIG["device"])
                targets = batch['target'].to(CONFIG["device"])
                
                with torch.amp.autocast(device_type='cuda'):
                    predictions = model(input_ids, attention_mask, image, ipq)
                    loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                all_targets_log.extend(targets.cpu().numpy()); all_preds_log.extend(predictions.cpu().numpy())
                
        local_smape = calculate_smape(np.expm1(all_targets_log), np.expm1(all_preds_log))
        print(f"Epoch {epoch+1}: Train Loss={(train_loss/len(train_loader)):.4f}, Val Loss={(val_loss/len(val_loader)):.4f}, 💡 Local SMAPE={local_smape:.4f}")
        
        scheduler.step()
        
        # 7. Early stopping logic
        if local_smape < best_smape_score:
            best_smape_score = local_smape
            torch.save(model.state_dict(), "../best_model.pth")
            print(f"New best model saved with SMAPE: {best_smape_score:.4f}!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"SMAPE did not improve for {epochs_no_improve} epoch(s). Best is {best_smape_score:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print(f"\nFine-tuning finished. Best validation SMAPE: {best_smape_score:.4f}")

if __name__ == "__main__":
    run_training()
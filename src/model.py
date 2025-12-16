import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class ProductPricePredictor(nn.Module):
    def __init__(self, text_model, image_model, dropout):
        super().__init__()

        # Text encoder
        self.text_model = AutoModel.from_pretrained(text_model)
        text_dim = self.text_model.config.hidden_size

        # Image encoder
        self.image_model = timm.create_model(
            image_model,
            pretrained=True,
            num_classes=0
        )
        image_dim = self.image_model.num_features

        # IMPORTANT: name MUST be regressor_head
        self.regressor_head = nn.Sequential(
            nn.LayerNorm(text_dim + image_dim + 1),
            nn.Linear(text_dim + image_dim + 1, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, image, ipq):
        text_feat = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        image_feat = self.image_model(image)

        x = torch.cat([text_feat, image_feat, ipq], dim=1)
        return self.regressor_head(x).squeeze(-1)

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch import nn

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load tokenizer and model from Rafal Poswiata's repo
model_name = "rafalposwiata/deproberta-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.to(device)
print("DepRoBERTa loaded successfully!")

# Add predictive/regression head
class DepRoBERTaPredictive(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.regressor = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_output =_


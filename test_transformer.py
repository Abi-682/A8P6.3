import torch
import torch.nn as nn

# Test transformer
model = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
x = torch.randn(2, 128, 32)
out = model(x)
print("Transformer ok")
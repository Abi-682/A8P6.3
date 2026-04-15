#!/usr/bin/env python3
"""
Motor Current Anomaly Detection
Problem 6.3: Compare LSTM, 1D CNN, and Transformer Encoder for time-series anomaly detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def generate_motor_current_data(n_samples=400, seq_len=128):
    """
    Generate synthetic motor current waveforms.
    Returns sequences (1200, 128), labels (1200,), class_names
    """
    np.random.seed(42)
    torch.manual_seed(42)

    sequences = []
    labels = []

    # Healthy: clean sinusoidal with noise and load variation
    for _ in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len)
        signal = np.sin(t) + 0.1 * np.random.randn(seq_len) + 0.05 * np.random.randn()
        sequences.append(signal)
        labels.append(0)

    # Bearing wear: add subtle high-frequency ripple
    for _ in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len)
        signal = np.sin(t) + 0.1 * np.random.randn(seq_len) + 0.05 * np.random.randn()
        ripple = 0.05 * np.sin(10*t)
        signal += ripple
        sequences.append(signal)
        labels.append(1)

    # Winding fault: unequal peaks (asymmetric)
    for _ in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len)
        signal = np.sin(t) + 0.1 * np.random.randn(seq_len) + 0.05 * np.random.randn()
        asymmetry = 0.02 * np.sin(2*t)
        signal += asymmetry
        sequences.append(signal)
        labels.append(2)

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    class_names = ["healthy", "bearing_wear", "winding_fault"]

    return sequences, labels, class_names


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h.squeeze(0))
        return out


class CNN1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, 1, 128)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1, 32)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, 32))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 3)
        self.attention_weights = None

        # Hook to capture attention from first layer
        def hook(module, input, output):
            self.attention_weights = output[1]

        self.encoder.layers[0].self_attn.register_forward_hook(hook)

    def forward(self, x, return_attention=False):
        x_proj = self.input_proj(x)
        x_pos = x_proj + self.pos_enc

        x = self.encoder(x_pos)
        x = x.mean(dim=1)
        out = self.fc(x)
        if return_attention:
            return out, self.attention_weights
        return out


def train_model(model, train_loader, val_loader, epochs=5, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}, Val Acc: {val_acc:.4f}")

    return model


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["healthy", "bearing_wear", "winding_fault"])
    return acc, report


def plot_waveforms(sequences, labels, class_names):
    plt.figure(figsize=(12, 4))
    for i, name in enumerate(class_names):
        idx = np.where(labels == i)[0][0]
        plt.subplot(1, 3, i+1)
        plt.plot(sequences[idx])
        plt.title(f"{name.replace('_', ' ').title()}")
        plt.xlabel("Time Step")
        plt.ylabel("Current")
    plt.tight_layout()
    plt.savefig("waveform_visualization.png")
    plt.close()


def plot_attention_heatmaps(model, sequences, labels, device='cpu'):
    model.eval()
    # Find one bearing_wear and one winding_fault
    bearing_idx = np.where(labels == 1)[0][0]
    winding_idx = np.where(labels == 2)[0][0]

    for idx, name in [(bearing_idx, "Bearing Wear"), (winding_idx, "Winding Fault")]:
        x = torch.tensor(sequences[idx:idx+1], dtype=torch.float32).unsqueeze(-1).to(device)
        with torch.no_grad():
            _, attn_weights = model(x, return_attention=True)
        # attn_weights shape: (seq_len, seq_len, batch, num_heads) wait, no
        # For MultiheadAttention, attn_weights is (batch, num_heads, seq_len, seq_len) if batch_first? Wait
        # Actually, in PyTorch, for MultiheadAttention, when need_weights=True, returns (batch, num_heads, seq_len, seq_len) if batch_first=True? Wait, let's check.
        # The doc says: attn_output, attn_weights = multihead_attn(query, key, value, need_weights=True)
        # attn_weights shape: (batch, num_heads, seq_len, seq_len) if batch_first is True for the module, but since we called self_attn directly, and self_attn is MultiheadAttention with batch_first=False by default? Wait.
        # nn.MultiheadAttention has batch_first parameter.
        # In TransformerEncoderLayer, it's created with batch_first=batch_first.
        # Since I set batch_first=True, self_attn has batch_first=True.
        # So attn_weights should be (batch, num_heads, seq_len, seq_len)
        attn = attn_weights[0].mean(dim=0).cpu().numpy()  # Average over heads

        plt.figure(figsize=(6, 5))
        plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.title(f"Self-Attention Weights: {name}")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.colorbar()
        plt.savefig(f"attention_{name.lower().replace(' ', '_')}.png")
        plt.close()


def main():
    # Generate data
    sequences, labels, class_names = generate_motor_current_data()
    print(f"Data shape: {sequences.shape}, Labels shape: {labels.shape}")

    # Visualize
    plot_waveforms(sequences, labels, class_names)

    # Prepare data
    sequences_reshaped = sequences.reshape(-1, 128, 1)  # For LSTM/Transformer
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences_reshaped, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val  # 15/85 ≈ 0.1765
    )

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    models = {
        "LSTM": LSTMModel(),
        "1D CNN": CNN1DModel(),
        "Transformer": TransformerModel()
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_model = train_model(model, train_loader, val_loader, device=device)
        test_acc, report = evaluate_model(trained_model, test_loader, device=device)
        results[name] = {"accuracy": test_acc, "report": report}
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        print(report)

    # Attention visualization for Transformer
    print("\nVisualizing attention for Transformer...")
    transformer = models["Transformer"]
    plot_attention_heatmaps(transformer, sequences, labels, device=device)

    # Save results for report
    with open("model_results.txt", "w") as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 40 + "\n")
        for name, res in results.items():
            f.write(f"{name} Test Accuracy: {res['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(res['report'] + "\n")
            f.write("-" * 40 + "\n")


if __name__ == "__main__":
    main()
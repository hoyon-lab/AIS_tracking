import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.va_vae_model import VAVAE, vae_loss
class AISTrackDataset(Dataset):
    def __init__(self, csv_path, min_len=50, fixed_len=50):
        df = pd.read_csv(csv_path)
        feature_cols = ["LAT", "LON", "SOG", "COG", "Heading"]
        mmsi_groups = df.groupby("MMSI")
        self.sequences = []
        for mmsi, group in mmsi_groups:
            group = group.sort_values(by="BaseDateTime")
            if len(group) < min_len:
                continue
            data = group[feature_cols].values
            for i in range(0, len(data) - fixed_len + 1):
                seq = data[i:i+fixed_len]
                self.sequences.append(seq)
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]  
def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss, total_recon, total_kl = 0, 0, 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(x_recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Recon: {total_recon:.4f} | KL: {total_kl:.4f}")
if __name__ == "__main__":
    csv_path = "./../dataset/AIS_2024_01_01/filtered_ais_nyc_classA_tracks.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AISTrackDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = VAVAE(input_dim=5, hidden_dim=64, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"🧠 Using device: {device}")
    print(f"📦 Loaded {len(dataset)} sequences.")
    train(model, dataloader, optimizer, device, epochs=10)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/va_vae_model.pt")

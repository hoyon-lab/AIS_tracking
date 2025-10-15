import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.va_vae_model import VAVAE, vae_loss
import os
import csv
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
def train(model, dataloader, optimizer, device, start_epoch, total_epochs, log_path):
    model.train()
    with open(log_path, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Epoch", "TotalLoss", "ReconLoss", "KLLoss"])
        for epoch in range(start_epoch, total_epochs + 1):
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
            print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | Recon: {total_recon:.4f} | KL: {total_kl:.4f}")
            writer.writerow([epoch, total_loss, total_recon, total_kl])
            if epoch % 10 == 0 or epoch == total_epochs:
                torch.save(model.state_dict(), f"checkpoints/va_vae_epoch_{epoch}.pt")
if __name__ == "__main__":
    csv_path = "./../dataset/AIS_2024_01_01/filtered_ais_nyc_classA_tracks.csv"
    model_path = "checkpoints/va_vae_model.pt"
    log_path = "logs/vae_train_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AISTrackDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = VAVAE(input_dim=5, hidden_dim=64, latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"🚀 재학습 시작: Epoch 11 → 300 | device: {device}")
    train(model, dataloader, optimizer, device, start_epoch=11, total_epochs=300, log_path=log_path)

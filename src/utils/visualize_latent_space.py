import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.va_vae_model import VAVAE
def load_sequences(csv_path, min_len=50, fixed_len=50):
    df = pd.read_csv(csv_path)
    feature_cols = ["LAT", "LON", "SOG", "COG", "Heading"]
    mmsi_groups = df.groupby("MMSI")
    sequences = []
    mmsi_labels = []
    for mmsi, group in mmsi_groups:
        group = group.sort_values(by="BaseDateTime")
        if len(group) < min_len:
            continue
        data = group[feature_cols].values
        for i in range(0, len(data) - fixed_len + 1):
            seq = data[i:i + fixed_len]
            sequences.append(seq)
            mmsi_labels.append(mmsi)
    return np.array(sequences), np.array(mmsi_labels)
def extract_latents(model, sequences, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in torch.tensor(sequences, dtype=torch.float32).to(device).split(256):
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            latents.append(z.cpu().numpy())
    return np.vstack(latents)
def plot_tsne(latents, mmsi_labels):
    print("🔄 Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(latents)
    plt.figure(figsize=(10, 8))
    unique_mmsi = np.unique(mmsi_labels)
    colors = plt.cm.get_cmap("tab20", len(unique_mmsi))
    mmsi_to_color = {mmsi: colors(i) for i, mmsi in enumerate(unique_mmsi)}
    for i, mmsi in enumerate(unique_mmsi):
        idx = mmsi_labels == mmsi
        plt.scatter(z_tsne[idx, 0], z_tsne[idx, 1], color=mmsi_to_color[mmsi], label=str(mmsi), s=10)
    plt.title("Latent Space (t-SNE Projection)")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    csv_path = "./../dataset/AIS_2024_01_01/filtered_ais_nyc_classA_tracks.csv"
    model_path = "checkpoints/va_vae_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📥 Loading sequences...")
    sequences, mmsi_labels = load_sequences(csv_path)
    model = VAVAE(input_dim=5, hidden_dim=64, latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("📦 Extracting latents...")
    latents = extract_latents(model, sequences, device)
    plot_tsne(latents, mmsi_labels)

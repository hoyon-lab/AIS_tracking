import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.va_vae_model import VAVAE  
model_path = "checkpoints/va_vae_epoch_300.pt"
csv_path = "./../dataset/AIS_2024_01_01/filtered_ais_nyc_classA_tracks.csv"
def load_sequences_with_labels(csv_path, min_len=50, fixed_len=50):
    df = pd.read_csv(csv_path)
    feature_cols = ["LAT", "LON", "SOG", "COG", "Heading"]
    mmsi_groups = df.groupby("MMSI")
    sequences, labels = [], []
    for mmsi, group in mmsi_groups:
        group = group.sort_values(by="BaseDateTime")
        if len(group) < min_len:
            continue
        data = group[feature_cols].values
        for i in range(0, len(data) - fixed_len + 1):
            sequences.append(data[i:i+fixed_len])
            labels.append(mmsi)
    return torch.tensor(sequences, dtype=torch.float32), labels
def extract_latents(model, sequences, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in sequences.split(256):
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            latents.append(z.cpu().numpy())
    return np.vstack(latents)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAVAE(input_dim=5, hidden_dim=64, latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    sequences, labels = load_sequences_with_labels(csv_path)
    MAX_SAMPLES = 3000
    if len(sequences) > MAX_SAMPLES:
        idx = np.random.choice(len(sequences), MAX_SAMPLES, replace=False)
        sequences = sequences[idx]
        labels = np.array(labels)[idx]
    latents = extract_latents(model, sequences, device)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(latents)
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))
    label_color = {m: colors(i % 20) for i, m in enumerate(unique_labels)}
    for m in unique_labels:
        idx = np.array(labels) == m
        plt.scatter(z_tsne[idx, 0], z_tsne[idx, 1], color=label_color[m], s=10, label=str(m))
    plt.title("Latent Space (epoch 300, t-SNE Projection)")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.tight_layout()
    plt.show()

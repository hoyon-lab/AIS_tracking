import torch
import torch.nn as nn
import torch.nn.functional as F
class VAVAE(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, latent_dim=32): 
        super(VAVAE, self).__init__()
        self.input_dim  = input_dim 
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    def encode(self, x):
        _, h = self.encoder_rnn(x)  
        h = h.squeeze(0)            
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, seq_len):
        h0 = self.decoder_input(z).unsqueeze(0)  
        B = z.size(0)
        decoder_inputs = torch.zeros((B, seq_len, self.input_dim)).to(z.device)  
        outputs, _ = self.decoder_rnn(decoder_inputs, h0)
        return self.output_layer(outputs)  
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
def train(model, dataloader, optimizer, device):
    model.train()
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
    print(f"Loss: {total_loss:.4f} | Recon: {total_recon:.4f} | KL: {total_kl:.4f}")

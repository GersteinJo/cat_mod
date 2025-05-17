import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (B, 1, 28, 28) -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 32, 14, 14) -> (B, 64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 64, 7, 7) -> (B, 128, 4, 4)
            nn.ReLU()
        )

        
        self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)  # Mean
        self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)  # Log-variance

        # Decoder (the reverse process of the encoder)
        self.fc3 = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # (B, 128, 4, 4) -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # (B, 32, 16, 16) -> (B, 1, 28, 28)
            nn.Sigmoid()  
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  
        mu = self.fc1(x)           
        logvar = self.fc2(x)       
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from the distribution
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)    
        return mu + eps * std          

    def decode(self, z):
        z = self.fc3(z) 
        z = z.view(z.size(0), 128, 4, 4)  
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# Loss function: Reconstruction loss + KL divergence
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross-Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence (Kullback-Leibler divergence)
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KL


def train(
    model,
    optimizer,
    train_loader,
    num_epochs,
    criterion = loss_function,
    device = 'cpu'
):
    train_loss_rec = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)  
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset)}")
        train_loss_rec.append(train_loss / len(train_loader.dataset))
        plt.plot(np.arange(epoch+1), train_loss_rec)


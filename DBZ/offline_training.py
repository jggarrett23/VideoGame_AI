from models import PreTrained_DeiTModel
import h5py
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*8*8, 512)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv_layers(X)
        X = self.flatten(X)
        return self.fc(X)


class ConvDecoder(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(latent_channels, 256 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (256, 8, 8))

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.fc(X)
        X = self.unflatten(X)
        return self.deconv_layers(X)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_in_channels, decoder_out_channels):
        super().__init__()

        self.encoder = ConvEncoder(
            in_channels=encoder_in_channels
        )

        self.decoder = ConvDecoder(
            latent_channels=512,
            out_channels=decoder_out_channels
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_enc = self.encoder(X)
        X_hat = self.decoder(X_enc)

        return X_hat

torch.manual_seed(23)

model_dir = './Models/'
check_point_dir = model_dir + 'checkpoints/'
model_name = "ViT_nLayers_4_nHeads_4_patch_6_emb_1024_20250117-200558"

datafile = model_name + '.h5'
data_file_path = './Gameplay_Data/' + datafile

with h5py.File(data_file_path, 'r') as f:
    states = f['state'][:] # M, 4, 128, 128
    actions = f['action'][:] # M,
    reward = f['reward'][:] # M, 
    next_states = f['next_state'][:] # M, 4, 128, 128

# learn low dimensional representation of each state
batch_size=64

# collapse state temporal dimension
states = states.reshape(-1, 1, *states.shape[2:])
states = torch.Tensor(states)

# normalize images
states = states / 255.

dataset = TensorDataset(states)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda')

model = AutoEncoder(
    encoder_in_channels=1,
    decoder_out_channels=1
).to(device)

optimizer = optim.NAdam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

nEpochs = 50
train_loss = []
val_loss = []


for epoch in trange(nEpochs):
    model.train()
    running_train_loss = 0
    for batch in train_loader:
        img = batch[0].to(device)

        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_loss.append(avg_train_loss)

    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            img = batch[0].to(device)

            recon = model(img)
            loss = criterion(recon, img)

            running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_loss.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{nEpochs}] | Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")

plt.semilogy(train_loss, label='Train Loss')
plt.semilogy(val_loss, label='Val Loss')
plt.legend()
plt.show()


test_img = next(iter(val_loader))[0][0].unsqueeze(0)

plt.figure()
plt.imshow(test_img.squeeze().detach().cpu().numpy())

model.eval()
with torch.no_grad():
    test_img_hat = model(test_img.to(device)).cpu().numpy()

plt.figure()
plt.imshow(test_img_hat.squeeze())

plt.show()


foo = 0
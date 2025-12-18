import streamlit as st
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Image Generation Comparison", layout="wide")
st.title("Autoencoder vs GAN vs Diffusion")
st.write("Upload a ZIP file containing 10 images of the same object in different environments.")

# ==============================
# DATASET
# ==============================
class ImageDataset(Dataset):
    def __init__(self, folder):
        self.image_paths = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# ==============================
# MODELS
# ==============================
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 3, 64, 64)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class DiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# ZIP UPLOAD
# ==============================
uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])

if uploaded_zip:
    if os.path.exists("images"):
        for root, dirs, files in os.walk("images", topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    else:
        os.mkdir("images")

    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall("images")

    dataset = ImageDataset("images")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    st.success(f"Loaded {len(dataset)} images")

    if st.button("Start Training (10 Epochs)"):

        # ==============================
        # AUTOENCODER TRAINING
        # ==============================
        ae = AutoEncoder()
        ae_opt = torch.optim.Adam(ae.parameters(), lr=0.001)
        mse = nn.MSELoss()

        for _ in range(10):
            for imgs in loader:
                loss = mse(ae(imgs), imgs)
                ae_opt.zero_grad()
                loss.backward()
                ae_opt.step()

        st.success("Autoencoder Training Completed")

        # ==============================
        # GAN TRAINING
        # ==============================
        G = Generator()
        D = Discriminator()
        g_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
        d_opt = torch.optim.Adam(D.parameters(), lr=0.0002)
        bce = nn.BCELoss()

        for _ in range(10):
            for real in loader:
                b = real.size(0)
                noise = torch.randn(b, 100)
                fake = G(noise)

                d_loss = bce(D(real), torch.ones(b, 1)) + \
                         bce(D(fake.detach()), torch.zeros(b, 1))
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                g_loss = bce(D(fake), torch.ones(b, 1))
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

        st.success("GAN Training Completed")

        # ==============================
        # DIFFUSION TRAINING
        # ==============================
        diff = DiffusionNet()
        diff_opt = torch.optim.Adam(diff.parameters(), lr=0.001)

        for _ in range(10):
            for imgs in loader:
                noise = torch.randn_like(imgs)
                noisy = imgs + noise
                loss = mse(diff(noisy), noise)
                diff_opt.zero_grad()
                loss.backward()
                diff_opt.step()

        st.success("Diffusion Training Completed")

        # ==============================
        # IMAGE GENERATION (FIXED)
        # ==============================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Autoencoder Output")
            img = ae(torch.randn(1, 3, 64, 64))
            img = img.squeeze().permute(1, 2, 0)
            img = ((img + 1) / 2).detach().cpu().numpy()
            st.image(img, clamp=True)

        with col2:
            st.subheader("GAN Output")
            img = G(torch.randn(1, 100))
            img = img.squeeze().permute(1, 2, 0)
            img = ((img + 1) / 2).detach().cpu().numpy()
            st.image(img, clamp=True)

        with col3:
            st.subheader("Diffusion Output")
            img = diff(torch.randn(1, 3, 64, 64))
            img = img.squeeze().permute(1, 2, 0)
            img = ((img + 1) / 2).detach().cpu().numpy()
            st.image(img, clamp=True)

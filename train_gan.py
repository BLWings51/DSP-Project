import argparse
import os
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess() -> Tuple[pd.DataFrame, List[str]]:
    """Load original data and run the same preprocessing pipeline used in the app."""
    from data_loader import load_transaction_data, preprocess_data  # local import to avoid circular deps

    raw_df = load_transaction_data()
    df_proc, _ = preprocess_data(raw_df)
    feature_cols = [c for c in df_proc.columns if c != "fraud"]
    return df_proc, feature_cols


def prepare_fraud_data(df_proc: pd.DataFrame,
                        feature_cols: List[str],
                        fraud_column: str = "fraud") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice out fraud rows, drop label, and scale each feature to [-1, 1]."""
    fraud_df = df_proc[df_proc[fraud_column] == 1].copy()
    X = fraud_df[feature_cols].values.astype(np.float32)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    scale = maxs - mins
    # avoid divide by zero for constant columns
    scale[scale == 0] = 1.0
    X_scaled = 2 * (X - mins) / scale - 1  # → [-1, 1]
    return X_scaled, mins, maxs


class Generator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Each feature already scaled to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAN:
    """Simple GAN wrapper that tracks losses and exposes sample() method."""

    def __init__(self, input_dim: int, output_dim: int,
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.generator = Generator(input_dim, output_dim).to(self.device)
        self.discriminator = Discriminator(output_dim).to(self.device)

        self.g_opt = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.g_losses: list[float] = []
        self.d_losses: list[float] = []
        self.input_dim = input_dim

    def _train_discriminator(self, real: torch.Tensor) -> float:
        batch_sz = real.size(0)
        noise = torch.randn(batch_sz, self.input_dim, device=self.device)
        fake = self.generator(noise).detach()

        # Labels
        real_labels = torch.ones(batch_sz, 1, device=self.device)
        fake_labels = torch.zeros(batch_sz, 1, device=self.device)

        # ---------------- Real loss
        real_pred = self.discriminator(real)
        d_real_loss = self.criterion(real_pred, real_labels)
        # ---------------- Fake loss
        fake_pred = self.discriminator(fake)
        d_fake_loss = self.criterion(fake_pred, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()
        return d_loss.item()

    def _train_generator(self, batch_sz: int) -> float:
        noise = torch.randn(batch_sz, self.input_dim, device=self.device)
        fake = self.generator(noise)
        labels = torch.ones(batch_sz, 1, device=self.device)  # want discriminator to predict real

        g_pred = self.discriminator(fake)
        g_loss = self.criterion(g_pred, labels)

        self.g_opt.zero_grad()
        g_loss.backward()
        self.g_opt.step()
        return g_loss.item()

    def train(self, data_array: np.ndarray, *,
              batch_size: int = 64, num_epochs: int = 1000, n_critic: int = 5):
        """Main training loop."""
        tensor_data = torch.tensor(data_array, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(num_epochs):
            d_epoch, g_epoch = 0.0, 0.0
            for i, (real_batch,) in enumerate(loader):
                d_loss = self._train_discriminator(real_batch)
                d_epoch += d_loss

                if i % n_critic == 0:
                    g_loss = self._train_generator(real_batch.size(0))
                    g_epoch += g_loss

            self.d_losses.append(d_epoch / len(loader))
            self.g_losses.append(g_epoch / (len(loader) / n_critic))

            if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1:04}/{num_epochs} | D: {self.d_losses[-1]:.4f} | G: {self.g_losses[-1]:.4f}")


    def sample(self, n: int) -> np.ndarray:
        """Generate *scaled* synthetic samples (still in [-1, 1] range)."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.input_dim, device=self.device)
            synth = self.generator(z).cpu().numpy()
        return synth


def main():
    parser = argparse.ArgumentParser(description="Pretrain a GAN for fraud synthesis")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--noise_dim", type=int, default=100, help="Latent noise vector length")
    parser.add_argument("--out_dir", type=str, default="models", help="Directory for saved artefacts")
    args = parser.parse_args()

    print("Loading and preprocessing data …")
    df_proc, feature_cols = load_and_preprocess()
    X_scaled, mins, maxs = prepare_fraud_data(df_proc, feature_cols)

    print(f" Fraud rows: {len(X_scaled):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gan = GAN(input_dim=args.noise_dim, output_dim=X_scaled.shape[1], device=device)

    print(" Training GAN …")
    gan.train(X_scaled, batch_size=args.batch_size, num_epochs=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    gen_path = os.path.join(args.out_dir, "gan_generator.pth")
    disc_path = os.path.join(args.out_dir, "gan_discriminator.pth")
    scale_path = os.path.join(args.out_dir, "gan_fraud_scaler.npz")

    torch.save(gan.generator.state_dict(), gen_path)
    torch.save(gan.discriminator.state_dict(), disc_path)
    np.savez(scale_path, mins=mins, maxs=maxs, feature_cols=np.array(feature_cols))

    print(" Saved artefacts:")
    print(" gen_path ", gen_path)
    print(" disc_path  ", disc_path)
    print(" scale_path  ", scale_path)
    print("Done!")


if __name__ == "__main__":
    main()

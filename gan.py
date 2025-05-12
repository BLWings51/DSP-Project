import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in range [-1, 1] for normalized data
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class GAN:
    def __init__(self, input_dim, output_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = Generator(input_dim, output_dim).to(device)
        self.discriminator = Discriminator(output_dim).to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Store losses for plotting
        self.g_losses = []
        self.d_losses = []

    def train(self, real_data, batch_size=64, num_epochs=1000, n_critic=5):
        """
        Train the GAN on fraud data.
        
        Parameters:
        -----------
        real_data : numpy.ndarray
            Normalized fraud data
        batch_size : int
            Batch size for training
        num_epochs : int
            Number of training epochs
        n_critic : int
            Number of discriminator updates per generator update
        """
        # Convert data to PyTorch tensors
        real_data = torch.FloatTensor(real_data).to(self.device)
        dataset = TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(num_epochs):
            for i, (real_batch,) in enumerate(dataloader):
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(real_batch.size(0), 1).to(self.device)
                d_real_output = self.discriminator(real_batch)
                d_real_loss = self.criterion(d_real_output, real_labels)
                
                # Fake data
                noise = torch.randn(real_batch.size(0), self.generator.input_dim).to(self.device)
                fake_batch = self.generator(noise)
                fake_labels = torch.zeros(real_batch.size(0), 1).to(self.device)
                d_fake_output = self.discriminator(fake_batch.detach())
                d_fake_loss = self.criterion(d_fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator every n_critic steps
                if i % n_critic == 0:
                    self.g_optimizer.zero_grad()
                    
                    # Generate fake data
                    noise = torch.randn(real_batch.size(0), self.generator.input_dim).to(self.device)
                    fake_batch = self.generator(noise)
                    g_output = self.discriminator(fake_batch)
                    g_loss = self.criterion(g_output, real_labels)
                    
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    # Store losses
                    self.g_losses.append(g_loss.item())
                    self.d_losses.append(d_loss.item())
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    def generate_samples(self, num_samples):
        """
        Generate synthetic fraud samples.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        numpy.ndarray
            Generated samples
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.input_dim).to(self.device)
            samples = self.generator(noise)
        return samples.cpu().numpy()

    def plot_losses(self):
        """Plot the training losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.title('GAN Training Losses')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def prepare_fraud_data(df, fraud_column='fraud'):
    """
    Prepare fraud data for GAN training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing transaction data
    fraud_column : str
        Name of the fraud indicator column
        
    Returns:
    --------
    numpy.ndarray
        Normalized fraud data
    """
    # Get fraud data
    fraud_data = df[df[fraud_column] == 1].drop(fraud_column, axis=1).values
    
    # Normalize data to [-1, 1] range
    fraud_data = 2 * (fraud_data - fraud_data.min()) / (fraud_data.max() - fraud_data.min()) - 1
    
    return fraud_data

def generate_synthetic_fraud(gan, num_samples, original_columns, fraud_column='fraud'):
    """
    Generate synthetic fraud samples and return them as a pandas DataFrame.
    
    Parameters:
    -----------
    gan : GAN
        Trained GAN model
    num_samples : int
        Number of synthetic samples to generate
    original_columns : list
        List of column names from the original DataFrame
    fraud_column : str, optional
        Name of the fraud indicator column. Defaults to 'fraud'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic fraud samples
    """
    try:
        # Generate synthetic samples
        synthetic_data = gan.generate_samples(num_samples)
        
        # Create DataFrame with original column names (excluding fraud column)
        df_columns = [col for col in original_columns if col != fraud_column]
        synthetic_df = pd.DataFrame(synthetic_data, columns=df_columns)
        
        # Add fraud column
        synthetic_df[fraud_column] = 1
        
        # Reorder columns to match original DataFrame
        synthetic_df = synthetic_df[original_columns]
        
        return synthetic_df
        
    except Exception as e:
        raise Exception(f"An error occurred while generating synthetic fraud samples: {str(e)}")

def save_synthetic_samples(synthetic_df, filename='synthetic_fraud.csv', directory='data'):
    """
    Save synthetic fraud samples to a CSV file.
    
    Parameters:
    -----------
    synthetic_df : pandas.DataFrame
        DataFrame containing synthetic fraud samples
    filename : str, optional
        Name of the output CSV file. Defaults to 'synthetic_fraud.csv'
    directory : str, optional
        Directory to save the file. Defaults to 'data'
        
    Returns:
    --------
    str
        Path to the saved file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Construct full file path
        filepath = os.path.join(directory, filename)
        
        # Save to CSV
        synthetic_df.to_csv(filepath, index=False)
        
        print(f"Successfully saved {len(synthetic_df)} synthetic samples to {filepath}")
        return filepath
        
    except Exception as e:
        raise Exception(f"Error saving synthetic samples: {str(e)}")

if __name__ == "__main__":
    from data_loader import load_transaction_data, preprocess_data
    
    # Load and preprocess data
    df = load_transaction_data()
    df_processed, _ = preprocess_data(df)
    
    # Prepare fraud data
    fraud_data = prepare_fraud_data(df_processed)
    
    # Initialize and train GAN
    input_dim = 100  # Size of noise vector
    output_dim = fraud_data.shape[1]  # Number of features
    gan = GAN(input_dim, output_dim)
    
    print("Training GAN...")
    gan.train(fraud_data, batch_size=64, num_epochs=1000)
    
    # Generate synthetic samples
    num_samples = 1000
    synthetic_df = generate_synthetic_fraud(
        gan,
        num_samples,
        original_columns=df_processed.columns.tolist()
    )
    
    print(f"\nGenerated {num_samples} synthetic fraud samples")
    print("\nSynthetic data summary:")
    print(synthetic_df.describe())
    
    # Save synthetic samples
    save_synthetic_samples(synthetic_df)
    
    # Plot training losses
    gan.plot_losses() 
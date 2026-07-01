import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt

# Configure matplotlib for IEEE-style academic plots
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12

###############################################################################
# 1. DATA LOADING AND ROBUST PREPROCESSING
###############################################################################

data_path = "MS_FA_labels_PASAT.csv"
df = pd.read_csv(data_path)

feature_cols = [c for c in df.columns if c != 'Label']
y = df['Label'].values
X = df[feature_cols].values

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)

###############################################################################
# 2. ROBUST VARIATIONAL AUTOENCODER (VAE)
###############################################################################

class RobustVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(RobustVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=0.05):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

model = RobustVAE(input_dim=X_scaled.shape[1], latent_dim=2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training Loop
print("--- Training Robust VAE ---")
epochs = 200
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch in dataloader:
        inputs = batch[0]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        loss = loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(dataloader):.4f}")

model.eval()
with torch.no_grad():
    _, latent_embeddings, _ = model(X_tensor)
latent_embeddings = latent_embeddings.numpy()

###############################################################################
# 3. GMM CLUSTERING & METRICS COMPUTATION
###############################################################################

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
cluster_preds = gmm.fit_predict(latent_embeddings)

# Compute alignment and evaluation metrics
ari = adjusted_rand_score(y, cluster_preds)
ami = adjusted_mutual_info_score(y, cluster_preds)

print("\n--- Unsupervised Geometry Metrics ---")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Adjusted Mutual Information (AMI): {ami:.4f}\n")

###############################################################################
# 4. STUDY 2: FEATURE IMPORTANCE VIA BACKPROP
###############################################################################
model.zero_grad()
X_tensor.requires_grad_(True)
mu_latent, _ = model.encode(X_tensor)
latent_energy = torch.sum(mu_latent**2)
latent_energy.backward()
feature_importance = torch.mean(torch.abs(X_tensor.grad), dim=0).detach().numpy()

top_indices = np.argsort(feature_importance)[::-1][:5]
top_features = [feature_cols[i] for i in top_indices]
top_scores = feature_importance[top_indices]

print("--- Study 2: Discovered Biomarkers (Top 5 ROI Tracts affecting Latent Space) ---")
for name, score in zip(top_features, top_scores):
    clean_name = name.replace("Left ", "L-").replace("Right ", "R-")[:40]
    print(f"- {clean_name}: {score:.4f}\n")

###############################################################################
# 5. SEPARATE VISUALIZATIONS (INDIVIDUAL PLOTS)
###############################################################################
labels_truth = ['HC-CP', 'RRMS-CP', 'RRMS-CI']

# --- FIGURE 1: Latent Space vs Clinical Phenotypes ---
plt.figure(figsize=(5, 4.5))
colors_truth = ['#2ca02c', '#1f77b4', '#d62728']

for i, color in enumerate(colors_truth):
    idx = (y == i)
    plt.scatter(latent_embeddings[idx, 0], latent_embeddings[idx, 1], 
                c=color, label=labels_truth[i], alpha=0.7, edgecolors='k', linewidth=0.5)
#plt.title(r'\textbf{VAE Latent Space Distribution}')
plt.xlabel(r'Latent Dimension 1 ($z_1$)')
plt.ylabel(r'Latent Dimension 2 ($z_2$)')
plt.legend(frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("fig1_latent_clinical_phenotypes.pdf", dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 2: Discovered GMM Densities ---
plt.figure(figsize=(5, 4.5))
colors_gmm = ['#bcbd22', '#17becf', '#9467bd']
for i in range(3):
    idx = (cluster_preds == i)
    plt.scatter(latent_embeddings[idx, 0], latent_embeddings[idx, 1], 
                c=colors_gmm[i], label=labels_truth[i], alpha=0.7, edgecolors='k', linewidth=0.5)
#plt.title(r'\textbf{Discovered GMM Clusters}')
plt.xlabel(r'Latent Dimension 1 ($z_1$)')
plt.ylabel(r'Latent Dimension 2 ($z_2$)')
plt.legend(frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("fig2_unsupervised_gmm_densities.pdf", dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 3: Feature Importance (Top ROI Drivers) ---
plt.figure(figsize=(5.5, 4.5))
short_names = [f.replace("Left ", "L-").replace("Right ", "R-")[:25] for f in top_features]
plt.barh(short_names[::-1], top_scores[::-1], color='#7f7f7f', edgecolor='k', height=0.5)
#plt.title(r'\textbf{Top Microstructural ROI Drivers}')
plt.xlabel(r'Mean Absolute Gradient ($\kappa$)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_feature_importance_biomarkers.pdf", dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 4: Continuous Trajectory Proof (Boxplot of Main Trajectory Component) ---
plt.figure(figsize=(5.5, 4.5))
box_data = [latent_embeddings[y == i, 0] for i in range(3)]
plt.boxplot(box_data, labels=labels_truth, patch_artist=True,
            boxprops=dict(facecolor='#e0e0e0', color='black'),
            medianprops=dict(color='red', linewidth=1.5))
#plt.title(r'\textbf{Microstructural Trajectory Alignment}')
plt.ylabel(r'Latent Projection Spectrum ($z_1$)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("fig4_trajectory_alignment_boxplot.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("Execution completed successfully. Metrics printed and 4 individual PDFs saved.")
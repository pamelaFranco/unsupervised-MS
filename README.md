# Unsupervised Deep Embedding of Fractional Anisotropy Landscapes for Differentiating Cognitive Phenotypes in Multiple Sclerosis

This repository contains the official **unsupervised deep representation and manifold learning pipeline** for mapping continuous neurocognitive degradation trajectories in Relapsing-Remitting Multiple Sclerosis (RRMS) using Fractional Anisotropy (FA) signatures from subcortical white matter compartments.

Instead of traditional supervised regression or rigid classification schemes, this framework reformulates cognitive phenotype characterization as an **unsupervised manifold embedding problem**. We leverage a regularized Variational Autoencoder (VAE) and a downstream Gaussian Mixture Model (GMM) to discover data-driven biological continuums directly from high-dimensional juxtacortical U-fiber networks.

---

##  Repository Contents

```text
├── Code/
│   └── ML_code.py             # Main VAE-GMM training, evaluation & gradient backpropagation script
├── Dataset/
│   └── MS_FA_labels_PASAT.csv # Processed cohort matrix (100 U-fiber-rich ROIs) and clinical labels
├── Figures/
│   ├── fig1.pdf               # Latent Space Continuous Phenotype Distribution
│   ├── fig2.pdf               # Discovered Unsupervised GMM Densities
│   ├── fig3.pdf               # Microstructural Trajectory Alignment (Boxplot)
│   └── fig4.pdf               # Top 5 Feature Importance Biomarkers (Backprop Gradients κ)
└── README.md                  # Project documentation and replication guide
# Unsupervised Deep Embedding of Fractional Anisotropy Landscapes for Differentiating Cognitive Phenotypes in Multiple Sclerosis

> **Note for Reviewers:** This repository hosts the official computational framework and reproducible workflows corresponding to the conference paper prepared for **6to Workshop Chileno sobre Reconocimiento de Patrones (CWPR 2026)**.


This repository contains the official **unsupervised deep representation and manifold learning pipeline** for mapping continuous neurocognitive degradation trajectories in Relapsing-Remitting Multiple Sclerosis (RRMS) using Fractional Anisotropy (FA) signatures from subcortical white matter compartments.

Instead of traditional supervised regression or rigid classification schemes, this framework reformulates cognitive phenotype characterization as an **unsupervised manifold embedding problem**. We leverage a regularized Variational Autoencoder (VAE) and a downstream Gaussian Mixture Model (GMM) to discover data-driven biological continuums directly from high-dimensional juxtacortical U-fiber networks.

---

##  Repository Contents

```text
├── Code/
│   └── ML_code.py             # Main VAE-GMM training, evaluation & gradient backpropagation script
│   └── requirements.txt                               # Python package dependencies.
├── Dataset/
│   └── MS_FA_labels_PASAT.csv # Processed cohort matrix (100 U-fiber-rich ROIs) and clinical labels
├── Figures/
│   ├── fig1.pdf               # Latent Space Continuous Phenotype Distribution
│   ├── fig2.pdf               # Discovered Unsupervised GMM Densities
│   ├── fig3.pdf               # Microstructural Trajectory Alignment (Boxplot)
│   └── fig4.pdf               # Top 5 Feature Importance Biomarkers (Backprop Gradients κ)
└── README.md                  # Project documentation and replication guide
```

---

## Abstract

Characterizing subcortical white matter (WM) degradation in Relapsing-Remitting Multiple Sclerosis (RRMS) presents a high-dimensional neuroimaging challenge. Conventional clinical paradigms categorize cognitive impairment using static performance thresholds, failing to capture the underlying continuous spectrum of structural network disruption. 

In this study, we propose a non-linear unsupervised pattern recognition framework designed to map continuous neurocognitive degradation trajectories. Leveraging a regularized Variational Autoencoder (VAE), we project high-dimensional subcortical Fractional Anisotropy (FA) landscapes from short-range association fibers (U-fibers) into a continuous two-dimensional latent manifold ($\mathbf{z} \in \mathbb{R}^2$). Downstream spatial density profiling via an unsupervised Gaussian Mixture Model (GMM) uncovers highly overlapping clusters that naturally align with clinical phenotypes, mathematically substantiating the existence of a biological continuum ($ARI = 0.1407$, $AMI = 0.1431$). 

Furthermore, we enforce model transparency through backpropagation-based sensitivity analysis ($\kappa$), identifying localized microstructural degradation anchors within temporo-parietal and fronto-insular association channels. Our results demonstrate that data-driven latent trajectories successfully preserve the progressive vector of neurodegeneration, establishing a robust framework for observer-independent computer-aided diagnosis.

---

## Framework & Pipeline Overview

The complete pattern recognition pipeline is structured into four core phases:

1. **Multimodal Neuroimaging & Masking**: Diffusion Tensor Imaging (DTI) sequences ($15\text{ directions}, b = 1000\text{ s/mm}^2$) and 3D T1-weighted images are processed using FSL and DSI Studio. Following spatial normalization via SPM12, mean FA metrics are extracted from 100 U-fiber-rich regions delineated by the LNAO-SWM79 Atlas, yielding the input matrix $\mathbf{X} \in \mathbb{R}^{93 \times 100}$.
2. **Deep Latent Embedding via Regularized VAE**: Input matrices are normalized using outlier-robust scaling. A smooth 2D continuous bottleneck ($\mathbf{z} \in \mathbb{R}^2$) is enforced using an architecture optimized with a joint Mean Squared Error (MSE) reconstruction loss and a Kullback-Leibler Divergence (KLD) regularizer scaled by a weight factor $\beta = 0.05$.
3. **Unsupervised Spatial Density Mapping**: Following convergence over 200 epochs via the AdamW optimizer, the deterministic latent coordinates are subjected to a 3-component full-covariance GMM to outline data-driven density distributions without human supervisor constraints.
4. **Transparent Feature Attribution via Backpropagation**: Rather than treating the continuous manifold as a black box, feature importance ($\kappa_j$) for each subcortical tract is mathematically derived as the mean absolute gradient of the total latent energy with respect to the input layer:
$$\kappa_j = \frac{1}{N} \sum_{i=1}^N \left| \frac{\partial E_z}{\partial x_{i,j}} \right|$$

---

## Key Experimental Results

* **Manifold Alignment & Continuous Spectrum:** The data-driven clusters yield an Adjusted Rand Index (**ARI**) of **0.1407** and an Adjusted Mutual Information (**AMI**) of **0.1431**. This low-to-moderate alignment mathematically validates the presence of a highly overlapping biological continuum rather than artificial discrete partitions.
* **Discovered Microstructural Drivers:** Gradient backpropagation analysis ($\kappa$) isolates the top 5 localized subcortical anchors dictating the deformation of the latent space trajectory:
    1. **Left Medial orbitofrontal - Superior temporal** ($\kappa = 0.1275$)
    2. **Right Middle temporal - Supramarginal** ($\kappa = 0.1067$)
    3. **Right Inferior parietal - Inferior temporal** ($\kappa = 0.1057$)
    4. **Left Postcentral - Supramarginal** ($\kappa = 0.0961$)
    5. **Left Inferior parietal - Superior parietal** ($\kappa = 0.0955$)

---


## How to Run

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/pamelaFranco/unsupervised-MS.git](https://github.com/pamelaFranco/unsupervised-MS.git)
   cd unsupervised-MS
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the ML pipeline:  
   ```bash
   # Run machine learning pipeline
   python ML_code.py
   ```
  
---

## Reproducibility & Data Availability

* **Data Privacy Restrictions**: Individual structural connectivity matrices and raw neuroimaging data generated during this study are **not publicly available** due to clinical privacy restrictions imposed by the local ethics committee.
* **Clinical Inquiries**: For anonymized computational matrices or clinical inquiries, please contact **Ethel Ciampi (ethelciampi@gmail.com)**.


---
## Citation

```
@inproceedings{montalba2026unsupervised-ms-trajectories,
  title={Unsupervised Deep Embedding of Fractional Anisotropy Landscapes for Cognitive Phenotype Trajectory Mapping in Multiple Sclerosis},
  author={Montalba, Cristian and Franco, Pamela and Caulier-Cisterna, Ricardo and Cruz, Jos{\'e} P. and C{\'a}rcamo, Claudia and Andia, Marcelo E. and Ciampi, Ethel},
  booktitle={Proceedings of the Chilean Computer Science Days (JCC 2026) - 6th Chilean Workshop on Pattern Recognition (CWPR)},
  publisher={IEEE},
  year={2026},
  note={Submitted for publication / Under review}
}
```

---

## Acknowledgements

This work is supported by the National Agency for Research and Development (ANID), project ICN2021_004 of the Millennium Science Initiative Program. 

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
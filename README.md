Here is a professional, ready-to-use `README.md` file for the code provided. It is structured to reflect the academic and technical nature of the project.

***

# Neural SDE Framework for Climate-Adjusted Corporate Credit Risk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/repo/blob/main/Climate_Credit_SDE.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This repository contains the **Reproducibility Package** for a novel Neural Stochastic Differential Equation (SDE) framework designed to model corporate credit risk under climate transition scenarios.

Traditional structural models (e.g., Merton) rely on static volatility assumptions and discrete time-steps. This framework utilizes **Variational Autoencoders (VAEs)** coupled with **Neural SDEs** to model the latent evolution of a firm's health in continuous time, allowing for dynamic stress testing against climate transition shocks.

## 🚀 Key Features

*   **Neural SDE Architecture:** Combines a GRU-based encoder with a stochastic differential equation decoder ($dZ_t = \mu dt + \sigma dW_t$) to capture non-linear and stochastic credit dynamics.
*   **Climate Delta Sensitivity:** Introduces a quantifiable metric ($\Delta PD / \Delta \lambda$) to measure how sensitive specific sectors (Energy, Utilities vs. Tech) are to climate transition shocks.
*   **Robust Data Pipeline:** Includes a synthetic data generator that simulates realistic financial ratios and default paths, ensuring code reproducibility without reliance on unstable external APIs.
*   **End-to-End Implementation:** A self-contained workflow handling data generation, preprocessing, model training, SDE integration, and figure generation.

## 🛠️ Methodology

The model operates in three stages:

1.  **Latent Feature Extraction:** An `EncoderRNN` processes historical financial ratios (e.g., Debt-to-Assets, ROA) to predict the initial latent state distribution $\mathcal{N}(\mu, \sigma)$.
2.  **Stochastic Evolution:** A `DecoderSDE` simulates the continuous-time evolution of the firm's credit health latent variable $Z_t$ using a learnable drift and diffusion process.
3.  **Default Determination:** The Probability of Default (PD) is calculated based on the probability of the simulated path $Z_t$ hitting a specific default barrier, passed through a sigmoid function.

### Climate Shock Injection
To calculate the **Climate Delta**, we perturb the drift function of the SDE:
$$ \mu_{shocked}(Z_t, t) = \mu_{base}(Z_t, t) - \lambda \cdot \mathbb{I}_{risk} $$
Where $\lambda$ represents the intensity of a climate policy shock (e.g., carbon tax introduction).

## 📦 Installation & Usage

### Option 1: Google Colab (Recommended)
The easiest way to reproduce the results is to open the notebook in Google Colab. No local setup is required.

### Option 2: Local Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/climate-sde-credit.git
    cd climate-sde-credit
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchsde yfinance pandas numpy matplotlib seaborn scikit-learn signatory
    ```

3.  Run the script:
    ```bash
    python main.py
    ```

## 📊 Configuration

The `Config` class at the top of the script allows you to control the experiment:

```python
class Config:
    USE_SYNTHETIC_DATA = True  # Set to False to attempt fetching real data via yfinance
    EPOCHS = 80                # Training duration
    LATENT_DIM = 3             # Size of the SDE latent space
    LAMBDA_SHOCK = 0.1         # Magnitude of climate stress test
    # ...
```

## 📈 Results

The framework generates three key figures upon execution:

1.  **ELBO Loss Dynamics:** Visualizes the convergence of the Variational Lower Bound during training.
2.  **ROC Curve:** Demonstrates the model's ability to distinguish between default and non-default firms on the test set.
3.  **Climate Delta by Sector:** A bar chart quantifying the marginal increase in PD for High-Risk (Energy, Materials) vs. Low-Risk (Tech, Healthcare) sectors under climate stress.

## 📂 Project Structure

```
├── main.py              # The complete end-to-end script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── results/             # (Generated) Output figures
```

## ⚠️ Note on Data

By default, `USE_SYNTHETIC_DATA = True` is enabled. This generates plausible financial time-series data to ensure the code runs successfully for all users, as public financial APIs (like `yfinance`) can be rate-limited or have missing data for specific historical periods required by the complex lag structure of the model.

## 🤝 Citation

If you use this code in your research, please cite:



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

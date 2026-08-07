from pathlib import Path
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.models.mt.mte import TrafficAttackPredictor


custom_rc = {
    "figure.titlesize": 22,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "Arial",
    "legend.title_fontsize": 14,
    "legend.fontsize": 12,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
}


def apply_custom_theme() -> None:
    """Apply consistent Matplotlib styling."""
    mpl.rcParams.update(custom_rc)
    sns.set_style("whitegrid")


def plot_latent_space(
    country: str,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    model: Union[TabularAE, TabularVAE, TrafficAttackPredictor],
    device: str,
    max_samples: int = 1000,
    folder: Path = Path.cwd(),
    fname: str = "plot_latent_space.png",
    show: bool = False,
) -> None:
    """Scatter plot showing latent space using PCA/t-SNE."""
    apply_custom_theme()
    
    if len(X_cont) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_cont), max_samples, replace=False)
        X_cont = X_cont[indices]
        X_cat = X_cat[indices]
        print(f"[INFO] Using random subset of {max_samples} samples for latent space")

    Xc_tensor = torch.from_numpy(X_cont).to(device)
    Xk_tensor = torch.from_numpy(X_cat).to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, TabularAE):
            z = model.encode(Xc_tensor, Xk_tensor).cpu().numpy()
        elif isinstance(model, TabularVAE):
            z = model.encode_to_latent(Xc_tensor, Xk_tensor, False).cpu().numpy()
        else:
            z = model.encoder(Xc_tensor, Xk_tensor).cpu().numpy()
    if len(z) < 10:
        print(f"[INFO] Not enough samples for latent space visualization: {len(z)}")
        return
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    
    if len(z) >= 50:
        tsne = TSNE(n_components=2, perplexity=min(30, len(z)-1), random_state=42)
        z_tsne = tsne.fit_transform(z)
    
    fig, axes = plt.subplots(1, 3 if len(z) >= 50 else 2, figsize=(18, 6))
    
    axes[0].scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6, s=20)
    axes[0].set_title(f"{country} — Latent Space (PCA)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(z, showfliers=False)
    axes[1].set_title(f"{country} — Latent Dimension Distributions")
    axes[1].set_xlabel("Latent Dimension")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, axis='y', alpha=0.3)
    
    if len(z) >= 50:
        axes[2].scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.6, s=20)
        axes[2].set_title(f"{country} — Latent Space (t-SNE)")
        axes[2].set_xlabel("t-SNE 1")
        axes[2].set_ylabel("t-SNE 2")
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f"{country} — Latent Space Analysis", fontsize=16)
    plt.tight_layout()
    
    pca_path = folder / f"{fname[:-4]}_pca_coords.csv"
    pd.DataFrame(z_pca, columns=['PC1', 'PC2']).to_csv(pca_path, index=False)
    
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved latent space visualization to {fname}")
    if show: plt.show()
    plt.close(fig)
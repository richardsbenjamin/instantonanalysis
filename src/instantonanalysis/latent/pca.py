from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

if TYPE_CHECKING:
    from typing import Callable, Iterator


def plot_pca(pca_res: dict, mask: np.ndarray, save_path: str) -> None:
    pca = pca_res["pca_model"]
    pca_transform = pca_res["pca_transform"]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_transform[:, 0], pca_transform[:, 1],
                        c=mask, cmap='viridis', alpha=0.6, s=2)
    plt.colorbar(scatter, label="Heatwave (1) / No heatwave (0)")
    plt.xlabel('PC1 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel('PC2 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[1]*100))
    plt.title('PCA: Heatwave vs. No Heatwave')
    plt.savefig(save_path, format="jpeg", dpi=300, bbox_inches="tight")
    plt.close()


def run_pca(x: np.ndarray, batch_size: int | None = None) -> dict:
    if batch_size is not None:
        pca = IncrementalPCA(n_components=2, batch_size=batch_size)
        pca_transform = pca.fit_transform(x)
    else:
        pca = PCA(n_components=2)
        pca_transform = pca.fit_transform(x)
    return {"pca_model": pca, "pca_transform": pca_transform}


def run_incremental_pca(
    make_batches: Callable[[], Iterator[np.ndarray]],
    n_components: int = 2,
) -> dict:
    pca = IncrementalPCA(n_components=n_components)
    for batch in make_batches():
        pca.partial_fit(batch)
    transforms = [pca.transform(batch) for batch in make_batches()]
    return {"pca_model": pca, "pca_transform": np.concatenate(transforms, axis=0)}

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from typing import Optional


def _subsample(
        x: np.ndarray,
        mask: np.ndarray,
        max_samples: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample rows so the metric stays tractable on big levels."""
    n = len(mask)
    if max_samples is None or n <= max_samples:
        return x, mask
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return x[idx], mask[idx]


def classifier_auc(
        x: np.ndarray,
        mask: np.ndarray,
        max_samples: Optional[int] = 200_000,
        n_splits: int = 5,
        seed: int = 42,
    ) -> float:
    """Cross-validated ROC-AUC of a logistic classifier separating the two
    classes (heatwave=1 / non-heatwave=0) from the latent vectors.

    A standardised logistic regression is scored with stratified k-fold CV;
    the mean AUC is a single comparable separability number per level.
    ~0.5 means no differentiation, →1.0 means strong differentiation.
    """
    x, mask = _subsample(x, mask, max_samples, seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, mask, scoring="roc_auc", cv=cv)
    return float(scores.mean())


def pc_separation(pca_transform: np.ndarray, mask: np.ndarray) -> float:
    """Separation of the two classes in PC1/PC2 space.

    Euclidean distance between class centroids, normalised by the pooled
    within-class standard deviation, so it is comparable across levels with
    different PC scales (a simple standardised effect size).
    """
    pcs = pca_transform[:, :2]
    a = pcs[mask == 1]
    b = pcs[mask == 0]
    centroid_dist = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
    pooled_std = np.sqrt(
        0.5 * (a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1))
    ).mean()
    if pooled_std == 0:
        return float("nan")
    return float(centroid_dist / pooled_std)

import numpy as np
import logging
from typing import Tuple, List, Dict


def cluster_embeddings(
    vectors: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 50,
    min_cluster_size: int = 40,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Cluster embeddings using UMAP for dimensionality reduction and HDBSCAN.

    Args:
        vectors: (N, D) numpy array of embeddings.
        n_components: Target dimensionality for UMAP.
        n_neighbors: UMAP ``n_neighbors`` parameter.
        min_cluster_size: HDBSCAN ``min_cluster_size`` parameter.
        random_state: Random seed for reproducibility.

    Returns:
        labels: (N,) cluster labels (-1 indicates noise).
        centroids: dict mapping cluster_id -> centroid vector (original D dims).
    """
    if vectors.shape[0] == 0:
        raise ValueError("No vectors provided for clustering.")

    try:
        import umap  # type: ignore
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    try:
        import hdbscan  # type: ignore
    except ImportError:
        raise ImportError("hdbscan is required: pip install hdbscan")

    logging.info(
        f"Running UMAP (n_components={n_components}, n_neighbors={n_neighbors}) on {vectors.shape[0]} vectors…"
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, vectors.shape[0] - 1),
        random_state=random_state,
    )
    reduced = reducer.fit_transform(vectors)

    logging.info(
        f"Running HDBSCAN (min_cluster_size={min_cluster_size}) on reduced vectors…"
    )
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    labels = clusterer.fit_predict(reduced)

    unique_labels = sorted(set(labels))
    centroids: Dict[int, np.ndarray] = {}
    for cid in unique_labels:
        if cid == -1:
            # Noise cluster; skip centroid calculation
            continue
        mask = labels == cid
        centroid = vectors[mask].mean(axis=0)
        centroids[int(cid)] = centroid
    logging.info(
        f"Clustering complete. {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters found (+ noise)."
    )
    return labels, centroids


def sample_texts_per_cluster(
    labels: np.ndarray,
    texts: List[str],
    max_samples: int = 8,
) -> Dict[int, List[str]]:
    """Return up-to ``max_samples`` texts per cluster for labeling.

    Args:
        labels: cluster labels for each text.
        texts: original texts (title+abstract) aligned with labels.
        max_samples: cap of samples per cluster.

    Returns:
        Dict mapping cluster_id -> list[str] sample texts.
    """
    samples: Dict[int, List[str]] = {}
    for idx, cid in enumerate(labels):
        if cid == -1:
            continue  # skip noise
        lst = samples.setdefault(int(cid), [])
        if len(lst) < max_samples:
            lst.append(texts[idx])
    return samples 
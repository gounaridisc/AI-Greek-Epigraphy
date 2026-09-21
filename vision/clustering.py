"""Visual clustering and filename-based cluster summaries."""

from __future__ import annotations

import re


def choose_cluster_count(features, max_k=8):
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(features) < 3:
        return 1, pd.DataFrame()

    rows = []
    for k in range(2, min(max_k, len(features) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(features)
        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(features, labels, metric="cosine"),
                "cluster_sizes": dict(pd.Series(labels).value_counts().sort_index()),
            }
        )

    scores_df = pd.DataFrame(rows)
    best_k = int(scores_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    return best_k, scores_df


def id_before_rotation(inscription_id: str) -> str:
    return re.sub(r"_Rotation_?\d+.*$", "", inscription_id)


def filename_prefix(inscription_id: str) -> str:
    base = id_before_rotation(inscription_id)
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] == "IG":
        return f"{parts[0]}_{parts[1]}"
    match = re.match(r"^[A-Za-z]+", base)
    if match:
        return match.group(0)
    return parts[0]


def cluster_embeddings(preprocessed_df, embeddings):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances

    n_components = min(50, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    best_k, scores_df = choose_cluster_count(embeddings_pca)
    if best_k <= 1:
        labels = np.zeros(len(preprocessed_df), dtype=int)
        centroids = embeddings_pca.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(embeddings_pca)
        centroids = kmeans.cluster_centers_

    results_df = preprocessed_df.copy()
    results_df["cluster"] = labels
    results_df["pca_x"] = embeddings_pca[:, 0]
    results_df["pca_y"] = embeddings_pca[:, 1] if embeddings_pca.shape[1] > 1 else 0.0
    results_df["base_inscription_id"] = results_df["inscription_id"].apply(id_before_rotation)
    results_df["filename_prefix"] = results_df["inscription_id"].apply(filename_prefix)

    distances = pairwise_distances(embeddings_pca, centroids)
    results_df["distance_to_centroid"] = [
        distances[i, int(cluster)] for i, cluster in enumerate(results_df["cluster"])
    ]

    centroid_examples = []
    for cluster_id in sorted(results_df["cluster"].unique()):
        cluster_examples = (
            results_df[results_df["cluster"] == cluster_id]
            .sort_values("distance_to_centroid")
            .head(3)
            .copy()
        )
        cluster_examples["rank_near_centroid"] = range(1, len(cluster_examples) + 1)
        centroid_examples.append(cluster_examples)
    centroid_examples_df = pd.concat(centroid_examples, ignore_index=True)
    return results_df, scores_df, centroid_examples_df, embeddings_pca


def add_umap_coordinates(results_df, embeddings_pca):
    try:
        import umap.umap_ as umap
    except ImportError:
        print("UMAP is not installed; skipping UMAP coordinates.")
        return results_df

    if len(results_df) < 3:
        return results_df

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(results_df) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings_pca)
    results_df = results_df.copy()
    results_df["umap_x"] = coords[:, 0]
    results_df["umap_y"] = coords[:, 1]
    return results_df


def make_cluster_summaries(results_df, centroid_examples_df):
    import pandas as pd

    rows = []
    for cluster_id in sorted(results_df["cluster"].unique()):
        cluster_df = results_df[results_df["cluster"] == cluster_id].copy()
        examples = centroid_examples_df[centroid_examples_df["cluster"] == cluster_id]
        rows.append(
            {
                "cluster": cluster_id,
                "image_count": len(cluster_df),
                "dominant_filename_prefixes": format_counts(cluster_df["filename_prefix"]),
                "dominant_image_folders": format_counts(cluster_df["image_folder"]),
                "centroid_examples": "; ".join(examples["filename"].tolist()),
                "common_rotation_group_ids": ", ".join(
                    cluster_df["base_inscription_id"].value_counts().head(5).index.tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def format_counts(series, top_n=5) -> str:
    counts = series.value_counts().head(top_n)
    return ", ".join(f"{idx}: {count}" for idx, count in counts.items())


"""Run the image preprocessing and visual clustering pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_config import build_layout, ensure_output_subdir, require_dataset
from vision.clustering import add_umap_coordinates, cluster_embeddings, make_cluster_summaries
from vision.embeddings import extract_vit_embeddings
from vision.preprocessing import collect_matched_images, make_preprocessing_checks, preprocess_all_images


def run_pipeline(args, layout) -> None:
    import numpy as np

    require_dataset(layout)
    output_dir = ensure_output_subdir(layout, "vision")

    images_df = collect_matched_images(layout, max_images=args.max_images)
    if images_df.empty:
        raise ValueError("No matched images found.")

    cleaned_images, preprocessed_df = preprocess_all_images(images_df)
    np.save(output_dir / "preprocessed_images.npy", cleaned_images)
    preprocessed_df.to_csv(output_dir / "preprocessed_images.csv", index=False, encoding="utf-8")

    checks_df = make_preprocessing_checks(preprocessed_df, cleaned_images)
    checks_df.to_csv(output_dir / "preprocessing_checks.csv", index=False, encoding="utf-8")

    embeddings = extract_vit_embeddings(cleaned_images, batch_size=args.batch_size)
    np.save(output_dir / "vit_embeddings.npy", embeddings)

    results_df, scores_df, centroid_examples_df, embeddings_pca = cluster_embeddings(preprocessed_df, embeddings)
    results_df = add_umap_coordinates(results_df, embeddings_pca)
    summaries_df = make_cluster_summaries(results_df, centroid_examples_df)

    results_df.to_csv(output_dir / "image_clusters.csv", index=False, encoding="utf-8")
    scores_df.to_csv(output_dir / "image_cluster_silhouette.csv", index=False, encoding="utf-8")
    centroid_examples_df.to_csv(output_dir / "cluster_centroid_examples.csv", index=False, encoding="utf-8")
    summaries_df.to_csv(output_dir / "cluster_summaries.csv", index=False, encoding="utf-8")

    print("Vision pipeline complete.")
    print(f"Matched images processed: {len(preprocessed_df):,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Clusters: {sorted(results_df['cluster'].unique())}")
    print(f"Outputs written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only the vision/image pipeline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--image-dirs", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args, build_layout(args))


if __name__ == "__main__":
    main()


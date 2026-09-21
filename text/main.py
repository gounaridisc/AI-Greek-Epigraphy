"""Run the text-processing and NLP pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_config import build_layout, ensure_output_subdir, find_valid_pairs, require_dataset
from text.analysis import (
    analyze_names,
    analyze_text_clusters,
    analyze_vocabulary,
    predict_aeneas_dates,
    prepare_aeneas_candidates,
)
from text.segmentation import build_vocabulary, deduplicate_segmented, make_segmenter
from text.transcription import load_annotations


def run_pipeline(args, layout) -> None:
    import pandas as pd

    require_dataset(layout)
    output_dir = ensure_output_subdir(layout, "text")

    valid_pairs = find_valid_pairs(layout)
    if not valid_pairs:
        raise ValueError("No matching annotation/image IDs found.")

    annotations_df = load_annotations(layout, valid_pairs)
    annotations_df.to_csv(output_dir / "annotations_converted.csv", index=False, encoding="utf-8")
    greek_transcripts = dict(zip(annotations_df["inscription_id"], annotations_df["greek_text"]))

    word_freq = build_vocabulary(max_rows=args.max_vocab_rows, skip_hf=args.skip_hf_vocab)
    segment_text = make_segmenter(word_freq)
    segmented = {inscription_id: segment_text(text) for inscription_id, text in greek_transcripts.items()}
    unique_segmented = deduplicate_segmented(segmented)

    segmented_df = pd.DataFrame(
        {
            "inscription_id": inscription_id,
            "segmented_text": " ".join(tokens),
            "token_count": len(tokens),
        }
        for inscription_id, tokens in segmented.items()
    )
    segmented_df.to_csv(output_dir / "segmented_transcripts.csv", index=False, encoding="utf-8")

    clean_words, _, vocab_summary, word_freq_df = analyze_vocabulary(unique_segmented)
    word_freq_df.to_csv(output_dir / "word_frequencies.csv", index=False, encoding="utf-8")
    pd.DataFrame([vocab_summary]).to_csv(output_dir / "vocabulary_summary.csv", index=False)

    names_df, name_freq_df, gender_summary_df = analyze_names(clean_words)
    names_df.to_csv(output_dir / "detected_names.csv", index=False, encoding="utf-8")
    name_freq_df.to_csv(output_dir / "name_frequencies.csv", index=False, encoding="utf-8")
    gender_summary_df.to_csv(output_dir / "name_gender_summary.csv", index=False, encoding="utf-8")

    clusters_df, silhouette_df, phrases_df, topics_df = analyze_text_clusters(unique_segmented, greek_transcripts)
    clusters_df.to_csv(output_dir / "text_clusters.csv", index=False, encoding="utf-8")
    silhouette_df.to_csv(output_dir / "text_cluster_silhouette.csv", index=False, encoding="utf-8")
    phrases_df.to_csv(output_dir / "cluster_repeated_phrases.csv", index=False, encoding="utf-8")
    topics_df.to_csv(output_dir / "text_topics_lda.csv", index=False, encoding="utf-8")

    aeneas_candidates_df = prepare_aeneas_candidates(unique_segmented, greek_transcripts)
    aeneas_candidates_df.head(10).to_csv(output_dir / "aeneas_candidate_texts.csv", index=False, encoding="utf-8")

    if getattr(args, "run_aeneas", False):
        aeneas_predictions_df = predict_aeneas_dates(
            aeneas_candidates_df,
            args.aeneas_repo_dir,
            args.aeneas_model_dir,
        )
        aeneas_predictions_df.to_csv(output_dir / "aeneas_predictions.csv", index=False, encoding="utf-8")

    print("Text pipeline complete.")
    print(f"Matched annotation/image pairs: {len(valid_pairs):,}")
    print(f"Converted annotations: {len(annotations_df):,}")
    print(f"Unique segmented inscriptions: {len(unique_segmented):,}")
    print(f"Vocabulary summary: {vocab_summary}")
    print(f"Detected proper-name tokens: {len(names_df):,}")
    print(f"Outputs written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only the text/NLP pipeline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--image-dirs", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-vocab-rows", type=int, default=None)
    parser.add_argument("--skip-hf-vocab", action="store_true")
    parser.add_argument("--run-aeneas", action="store_true")
    parser.add_argument("--aeneas-repo-dir", default="predictingthepast")
    parser.add_argument("--aeneas-model-dir", default="data/aeneas_model")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args, build_layout(args))


if __name__ == "__main__":
    main()


"""Run the OCR baseline, CER evaluation, and optional fine-tuning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr.crops import collect_ocr_images, prepare_gt_dir, prepare_line_crops, split_line_crops
from ocr.trocr import B3_BASELINE_MODEL_NAME, finetune_trocr, run_baseline
from pipeline_config import build_layout, ensure_output_subdir, require_dataset


def run_pipeline(args, layout) -> None:
    import pandas as pd

    require_dataset(layout)
    output_dir = ensure_output_subdir(layout, "ocr")
    eval_root = output_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    images_df = collect_ocr_images(layout, max_images=args.max_images)
    if images_df.empty:
        raise ValueError("No matched images found for OCR.")

    line_crops, lines_df = prepare_line_crops(images_df, layout.annotations_dir)
    if lines_df.empty:
        raise ValueError("No OCR line crops were extracted.")

    lines_df = split_line_crops(lines_df)
    lines_df.to_csv(output_dir / "line_crops_metadata.csv", index=False, encoding="utf-8")

    gt_dir, copied = prepare_gt_dir(lines_df, layout.annotations_dir, eval_root)
    print(f"Ground-truth files copied for official scoring: {copied}")

    baseline = run_baseline(lines_df, line_crops, eval_root, gt_dir, batch_size=args.batch_size)
    results = [
        {
            "method": "Baseline TrOCR-base",
            "model": B3_BASELINE_MODEL_NAME,
            "description": "Pretrained model, no post-processing, no fine-tuning",
            "CER": baseline["raw_cer"],
        },
        {
            "method": "Baseline TrOCR-base + normalization",
            "model": B3_BASELINE_MODEL_NAME,
            "description": "Uppercase, remove punctuation/spaces/non-proxy symbols",
            "CER": baseline["clean_cer"],
        },
    ]

    if args.finetune:
        _, ft_cer, model_dir = finetune_trocr(
            lines_df,
            line_crops,
            baseline["line_indices"],
            eval_root,
            gt_dir,
            output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(
            {
                "method": "Fine-tuned TrOCR-small",
                "model": str(model_dir),
                "description": "Fine-tuned on training line crops; encoder mostly frozen; normalized output",
                "CER": ft_cer,
            }
        )

    results_df = pd.DataFrame(results)
    baseline_cer = float(results_df.iloc[0]["CER"])
    results_df["absolute_improvement_vs_baseline"] = baseline_cer - results_df["CER"]
    results_df["relative_improvement_vs_baseline_%"] = (
        100 * (baseline_cer - results_df["CER"]) / baseline_cer if baseline_cer else 0.0
    )
    results_df.to_csv(output_dir / "ocr_results.csv", index=False, encoding="utf-8")

    print("OCR pipeline complete.")
    print(f"Line crops: {len(lines_df):,}")
    print(f"Baseline CER: {baseline['raw_cer']:.4f}")
    print(f"Normalized baseline CER: {baseline['clean_cer']:.4f}")
    print(f"Outputs written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only the OCR pipeline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--image-dirs", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args, build_layout(args))


if __name__ == "__main__":
    main()


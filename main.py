"""Run the refactored Greek papyri pipelines."""

from __future__ import annotations

import argparse

from pipeline_config import build_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run text, vision, and OCR pipelines for Greek papyri data."
    )
    parser.add_argument(
        "--only",
        choices=["all", "text", "vision", "ocr"],
        default="all",
        help="Run only one pipeline stage.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Dataset root. Expected layout is data/Annotations[/Annotations] and data/Images1..Images4.",
    )
    parser.add_argument(
        "--annotations-dir",
        default=None,
        help="Optional explicit annotation directory containing *_letters.txt files.",
    )
    parser.add_argument(
        "--image-dirs",
        default=None,
        help="Optional comma-separated image directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated CSVs, predictions, plots, and models.",
    )
    parser.add_argument(
        "--max-vocab-rows",
        type=int,
        default=None,
        help="Optional limit when streaming the Hugging Face vocabulary corpus.",
    )
    parser.add_argument(
        "--skip-hf-vocab",
        action="store_true",
        help="Use only the built-in seed vocabulary for text segmentation.",
    )
    parser.add_argument(
        "--run-aeneas",
        action="store_true",
        help="Run optional Aeneas dating attribution for the top text candidates.",
    )
    parser.add_argument(
        "--aeneas-repo-dir",
        default="predictingthepast",
        help="Local predictingthepast repository path for --run-aeneas.",
    )
    parser.add_argument(
        "--aeneas-model-dir",
        default="data/aeneas_model",
        help="Directory containing Aeneas Greek model files for --run-aeneas.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional image limit for quick vision/OCR runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size for ViT and TrOCR.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Run TrOCR-small fine-tuning in the OCR pipeline.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Fine-tuning epochs for --only ocr --finetune.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    layout = build_layout(args)

    if args.only in {"all", "text"}:
        from text.main import run_pipeline as run_text_pipeline

        run_text_pipeline(args, layout)

    if args.only in {"all", "vision"}:
        from vision.main import run_pipeline as run_vision_pipeline

        run_vision_pipeline(args, layout)

    if args.only in {"all", "ocr"}:
        from ocr.main import run_pipeline as run_ocr_pipeline

        run_ocr_pipeline(args, layout)


if __name__ == "__main__":
    main()

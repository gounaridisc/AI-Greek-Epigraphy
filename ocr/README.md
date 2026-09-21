# OCR and TrOCR Fine-Tuning for Greek Papyri

## Purpose

This directory contains the OCR component of the Greek papyri project. It works with papyri image files and matching letter-box annotation files, extracts line-level image crops, runs OCR on those crops, and evaluates predicted transcriptions against reference text.

OCR on Greek papyri is difficult because the source material is visually irregular: surfaces can be degraded, contrast can be weak, letters may be distorted or missing, and line geometry may not be perfectly horizontal. The transcription format also uses domain-specific conventions, including a Latin proxy alphabet for Greek characters and annotation files that combine geometry with text.

This module helps the full project by connecting the visual side of the pipeline to the textual side. It turns image regions into machine-readable OCR predictions, measures how close those predictions are to the reference transcriptions, and provides an optional path for adapting TrOCR to the project data.

## What This Module Does

The OCR module:

- finds images that have matching annotation files in the shared project layout;
- reads ICDAR-style letter-box annotation files;
- orders character boxes into text rows;
- deskews grayscale image regions using the annotation angle;
- cuts padded line crops from papyri images;
- creates train/test splits by base inscription ID;
- copies test-set ground-truth annotation files into an evaluation directory;
- runs baseline OCR inference with `microsoft/trocr-base-printed`;
- optionally normalizes OCR predictions before scoring;
- computes character error rate using Levenshtein distance;
- optionally fine-tunes `microsoft/trocr-small-printed` on training line crops;
- saves OCR predictions, CER summaries, line-crop metadata, fine-tuning losses, and the fine-tuned model.

## How It Works

1. The pipeline resolves the shared data layout and collects image records that have matching annotation files.
2. Each annotation file is parsed for rotation angle, character boxes, transcript characters, and optional line specifications.
3. Character boxes are ordered into rows, either from explicit line specifications or inferred from box geometry.
4. The source image is loaded in grayscale, deskewed, and cut into padded line crops.
5. Line crops are paired with their reference line text and saved as metadata.
6. Base inscription IDs are split into train and test groups so related rotations stay together.
7. Test ground-truth files are copied into `eval/gt/` for CER scoring.
8. The baseline TrOCR model predicts text for test line crops.
9. Raw predictions and normalized predictions are written as transcript files and evaluated with CER.
10. If `--finetune` is used, TrOCR-small is fine-tuned on training crops, evaluated on the same test lines, and saved.
11. A compact `ocr_results.csv` file records baseline, normalized baseline, and optional fine-tuned CER values.

## Files in This Directory

```text
main.py        Entry point for running the OCR subpipeline, baseline evaluation, and optional fine-tuning.
crops.py       Collects OCR images, extracts deskewed line crops, builds splits, and writes prediction files.
evaluation.py  Parses letter-box annotations, orders boxes into rows, reconstructs ground truth, and computes CER.
trocr.py       Loads TrOCR models, runs inference, normalizes predictions, and fine-tunes TrOCR-small.
__init__.py    Marks this directory as the OCR package.
README.md      Explains the purpose and structure of this OCR component.
```

## Methods and Technologies

- **OpenCV**: loads grayscale papyri images, applies affine deskewing, and converts crops for OCR.
- **NumPy**: handles box coordinates, geometric ordering, matrix transforms, and row inference.
- **Pillow**: converts image crops into RGB PIL images expected by the TrOCR processor.
- **pandas**: stores line-crop metadata and summary result tables.
- **scikit-learn**: creates train/test and fine-tuning validation splits by base inscription ID.
- **Hugging Face Transformers**: loads `TrOCRProcessor` and `VisionEncoderDecoderModel`.
- **TrOCR**: treats OCR as image-to-text sequence generation from cropped document lines.
- **PyTorch**: runs model inference and the optional fine-tuning loop.
- **Character Error Rate (CER)**: evaluates OCR quality at the character level using Levenshtein distance.
- **Prediction normalization**: uppercases, applies a few OCR confusion fixes, and removes non-letter characters before cleaned scoring.

## Inputs and Outputs

The OCR pipeline expects the shared project data layout used by the root pipeline. The important inputs are:

- image files discoverable through the configured image directories;
- annotation files named like `<inscription_id>_letters.txt`;
- annotation files containing sectioned data for rotation angle, character boxes, transcript text, and optional line specifications;
- inscription IDs that can be matched between image records and annotation files.

The annotation parser expects character boxes with eight numeric coordinates per box. For official-style CER scoring and prediction export, the current code only copies and scores test annotations whose IDs end with `_Rotation1_300dpi`.

Important command-line options include:

- `--data-dir` for the shared data root;
- `--annotations-dir` for annotation files;
- `--image-dirs` for image directories;
- `--output-dir` for generated OCR artifacts;
- `--max-images` to limit how many matched images are processed;
- `--batch-size` for TrOCR inference batch size;
- `--finetune` to enable optional model training;
- `--epochs` for the fine-tuning loop.

The module writes outputs under the OCR output directory, normally `outputs/ocr/`:

```text
line_crops_metadata.csv              Metadata for extracted line crops, including split labels.
ocr_results.csv                      CER summary for baseline, normalized baseline, and optional fine-tuning.
finetune_losses.csv                  Per-epoch training and validation loss, only when fine-tuning runs.
trocr-small-finetuned/               Saved fine-tuned model and processor, only when fine-tuning runs.
eval/gt/                             Copied ground-truth annotation files used for scoring.
eval/pred_baseline_trocr_base/       Raw baseline TrOCR transcript predictions.
eval/pred_baseline_trocr_base_cleaned/ Normalized baseline transcript predictions.
eval/pred_trocr_small_finetuned/     Normalized fine-tuned predictions, only when fine-tuning runs.
```

## Fine-Tuning

Fine-tuning is implemented but optional. It runs only when `main.py` is called with `--finetune`.

The fine-tuning path uses `microsoft/trocr-small-printed`, not the larger baseline model. It trains on line crops marked as `train`, then evaluates on the same test line indices used by the baseline. The code freezes the encoder by default, attempts to unfreeze the last two encoder layers, keeps the decoder trainable, enables gradient checkpointing when available, and trains with AdamW.

Training uses a small internal batch size of `1` with gradient accumulation. The `--batch-size` argument is used for inference, including fine-tuned test prediction. After training, the code saves:

- `trocr-small-finetuned/` with the model and processor;
- `finetune_losses.csv` with epoch-level training and validation loss;
- `eval/pred_trocr_small_finetuned/` with normalized test predictions;
- a fine-tuned row in `ocr_results.csv`.

The fine-tuning split is made from training base inscription IDs, with a small validation split. If there are not enough training base inscriptions, fine-tuning raises an error rather than silently training on an invalid split.

## Evaluation

Evaluation is based on character error rate. The evaluator reconstructs ground-truth transcript rows from annotation boxes, converts Greek characters into the same uppercase Latin proxy alphabet used by the project, strips spaces and tabs, and compares predictions with references using Levenshtein distance.

The baseline is evaluated twice:

- raw TrOCR output, saved under `pred_baseline_trocr_base`;
- normalized baseline output, saved under `pred_baseline_trocr_base_cleaned`.

Fine-tuned predictions are normalized before scoring. The final CER values are written to `ocr_results.csv` together with absolute and relative improvement versus the raw baseline. The code does not contain fixed benchmark numbers; results depend on the local data, model availability, and runtime configuration.

## Role in the Full Project

`ocr/` focuses on converting papyri image regions into text and measuring that conversion against reference transcriptions. It complements `vision/`, which preprocesses and analyzes images visually, and `text/`, which cleans, segments, and analyzes transcription text.

In the larger workflow, OCR outputs can be inspected as text artifacts, compared with reference annotations, and used to study where image quality, crop quality, or model behavior affects downstream textual analysis.

## Notes for Future Development

Useful next steps for this OCR component include:

- saving extracted line-crop images for easier visual inspection;
- adding richer prediction reports with per-line CER and error examples;
- making the `_Rotation1_300dpi` scoring assumption configurable;
- improving normalization rules for Greek proxy transcription conventions;
- adding image augmentation for fine-tuning;
- comparing additional OCR models or decoder settings;
- tracking experiment parameters beside `ocr_results.csv`;
- connecting OCR error patterns to visual clusters and text-analysis outputs.

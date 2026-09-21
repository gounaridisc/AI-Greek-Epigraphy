# Greek Papyri AI

Refactored Python pipelines from the Greek papyri / Greek squeezes OCR notebook (`Papyri.py`).

The project has three parts:

- `text/`: proxy-to-Greek conversion, scriptio continua word segmentation, vocabulary statistics, name/gender analysis, text clustering, topic modeling, and Aeneas-ready dating candidates.
- `vision/`: image loading, grayscale resizing/padding, CLAHE, denoising, ViT embeddings, PCA/UMAP coordinates, K-Means clustering, and cluster summaries.
- `ocr/`: line-crop preparation, TrOCR baseline inference, output normalization, CER evaluation, and optional TrOCR-small fine-tuning.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Data

Download the Greek squeezes / papyri OCR dataset manually and place it under `data/`.
See `data/README.md` for the expected folder layout.

## Run

Full pipeline:

```bash
python main.py
```

Run one part:

```bash
python main.py --only text
python main.py --only vision
python main.py --only ocr
python main.py --only ocr --finetune
```

Useful path overrides:

```bash
python main.py --only text --data-dir data
python main.py --only vision --annotations-dir data/Annotations/Annotations --image-dirs data/Images1,data/Images2,data/Images3,data/Images4
```

Outputs are written under `outputs/` by default.

`Papyri.py` is kept in the repository as the original converted notebook reference.


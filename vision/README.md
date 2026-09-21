# Visual Representation Learning for Greek Papyri Images

## Purpose

This subproject implements the computer-vision side of the larger Greek papyri analysis pipeline. It works with local papyri or Greek squeeze image files whose filename stems can be matched to annotation IDs. Its job is to make the image collection comparable: first by standardizing the raw images, then by extracting dense visual embeddings, and finally by clustering images with similar visual structure.

Ancient document images are difficult to analyze because the visual signal is often irregular. The images may contain damaged surfaces, faint letter strokes, uneven lighting, low contrast, scanning noise, irregular stone or paper shapes, and inconsistent image dimensions. These properties make direct pixel-level comparison weak and make downstream OCR behavior harder to interpret.

The `vision/` component contributes a visual representation layer to the full project. It does not perform OCR and it does not analyze Greek text. Instead, it prepares and summarizes the image side of the corpus so that visual similarity, image quality, repeated rotations, and collection-level patterns can be inspected alongside the text and OCR pipelines.

## What This Module Does

The `vision/` directory implements a complete image preprocessing and visual clustering pipeline:

- collects image files that have matching `*_letters.txt` annotation IDs,
- reads images from local image folders using OpenCV,
- converts raw images from BGR to RGB and then to grayscale,
- resizes images while preserving aspect ratio,
- pads images to a fixed `512 x 512` canvas,
- applies CLAHE local contrast enhancement with `clipLimit=1.5` and `tileGridSize=(8, 8)`,
- applies OpenCV non-local means denoising with `h=7`, `templateWindowSize=7`, and `searchWindowSize=21`,
- records preprocessing metadata and pixel-intensity checks,
- converts cleaned grayscale arrays into RGB PIL images for the transformer processor,
- extracts visual embeddings with the Hugging Face model `google/vit-base-patch16-224-in21k`,
- L2-normalizes the embedding matrix with scikit-learn,
- reduces embeddings with PCA,
- chooses a K-Means cluster count using cosine silhouette score,
- optionally adds UMAP coordinates when `umap-learn` is available,
- finds the three images closest to each cluster centroid,
- summarizes clusters using filename prefixes, image folders, and base IDs before `_Rotation`.

## How It Works

1. `main.py` receives the shared project layout, verifies that the dataset folders exist, and creates the `outputs/vision` directory.
2. `preprocessing.py` collects image records from configured image folders and keeps only images whose filename stem matches an annotation ID.
3. Each matched image is read with OpenCV, converted to grayscale, resized without distortion, padded to `512 x 512`, enhanced with CLAHE, and denoised.
4. The cleaned image tensor, preprocessing metadata, and pixel-intensity checks are saved for inspection.
5. `embeddings.py` converts the cleaned grayscale images into RGB PIL images and feeds them through `google/vit-base-patch16-224-in21k`.
6. The resulting ViT `[CLS]` embeddings are normalized and saved as a reusable NumPy matrix.
7. `clustering.py` applies PCA, evaluates candidate K-Means cluster counts, assigns final cluster labels, computes centroid distances, and creates compact cluster summaries.
8. `main.py` writes all outputs as `.npy` and `.csv` artifacts for later inspection or comparison with OCR behavior.

## Files in This Directory

```text
main.py             Entry point for running the vision subpipeline and writing outputs.
preprocessing.py    Handles image matching, loading, resizing, padding, CLAHE, denoising, and preprocessing checks.
embeddings.py       Extracts normalized Vision Transformer embeddings from cleaned images.
clustering.py       Runs PCA, K-Means, optional UMAP coordinates, centroid selection, and cluster summaries.
__init__.py         Marks this directory as a Python package.
README.md           Explains the purpose and structure of this vision component.
```

## Methods and Technologies

- **Python**: coordinates the visual analysis pipeline.
- **pandas**: stores image metadata, preprocessing records, cluster labels, centroid examples, and summaries.
- **NumPy**: stores the cleaned image tensor and embedding matrix.
- **OpenCV**: reads images, converts color spaces, resizes arrays, applies CLAHE, and performs denoising.
- **CLAHE**: improves local contrast in low-contrast or unevenly lit papyri images without relying on one global contrast adjustment.
- **Non-local means denoising**: reduces local image noise while trying to preserve stroke-like structures.
- **Pillow**: converts cleaned grayscale arrays into RGB PIL images required by the transformer image processor.
- **PyTorch**: runs the pretrained visual model on CPU or CUDA.
- **Hugging Face Transformers**: loads `AutoImageProcessor` and `AutoModel` for `google/vit-base-patch16-224-in21k`.
- **Vision Transformer embeddings**: convert each image into a dense visual representation that can be compared across the collection.
- **scikit-learn normalization**: normalizes embeddings so clustering depends more on vector direction than raw magnitude.
- **PCA**: reduces high-dimensional embeddings before clustering and stores two PCA coordinates for inspection.
- **K-Means**: groups images with similar visual embeddings.
- **cosine silhouette score**: selects a cluster count from candidate K values.
- **pairwise distances**: identifies images closest to each cluster centroid.
- **UMAP**: optionally creates two-dimensional coordinates for visual exploration if the package is installed.

## Inputs and Outputs

Inputs are configured through the shared CLI/path layout used by the root project:

- `--data-dir`: dataset root, defaulting to `data`.
- `--annotations-dir`: optional directory containing `*_letters.txt` annotation files.
- `--image-dirs`: optional comma-separated image directories.
- `--max-images`: optional limit for quick exploratory runs.
- `--batch-size`: batch size for ViT embedding extraction.

The accepted image extensions come from the shared path helper: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.bmp`. Images are matched to annotations by ID: an image file stem must match an annotation ID extracted from a `*_letters.txt` file. The vision code uses this match to keep the visual dataset aligned with the rest of the project, but it does not parse the annotation text.

Generated outputs are written under `outputs/vision` by default:

```text
preprocessed_images.npy          Cleaned grayscale image array.
preprocessed_images.csv          Per-image metadata and original/processed dimensions.
preprocessing_checks.csv         Mean, standard deviation, min, and max pixel-intensity checks.
vit_embeddings.npy               Normalized ViT embedding matrix.
image_clusters.csv               Cluster labels, PCA/UMAP coordinates, centroid distances, and metadata.
image_cluster_silhouette.csv     Candidate K values, silhouette scores, and cluster sizes.
cluster_centroid_examples.csv    Top images nearest to each cluster centroid.
cluster_summaries.csv            Cluster-level filename, folder, centroid, and rotation-group summaries.
```

The code has no hardcoded Colab paths. Paths are resolved through CLI arguments and `pipeline_config.py`.

## Role in the Full Project

`vision/` is responsible for the image-side exploration of the Greek papyri corpus. It prepares difficult historical document images, extracts visual structure with a pretrained transformer, and groups images by visual similarity.

It complements the other subprojects:

- `text/` analyzes the available transcriptions and linguistic structure.
- `ocr/` creates line crops, runs OCR inference, fine-tunes TrOCR when requested, and evaluates character error rate.
- `vision/` helps explain the visual variation that can affect OCR and corpus interpretation, such as image quality, repeated rotations, source-folder patterns, and visually similar document surfaces.

The outputs are especially useful for comparing image clusters with later OCR performance or for selecting representative examples from visually distinct parts of the dataset.

## Notes for Future Development

- Add saved visual plots for PCA and UMAP coordinates, since the current code stores coordinates and CSV summaries but does not render figures.
- Compare multiple preprocessing variants, such as different CLAHE settings, denoising strengths, or no-denoising baselines.
- Cache or reuse existing `vit_embeddings.npy` when the input image set has not changed.
- Add a small manifest that records model name, preprocessing constants, image count, and timestamp for each run.
- Try alternative visual backbones and compare their cluster stability.
- Connect image cluster assignments to OCR metrics from `ocr/` to see whether visual groups correspond to recognition difficulty.

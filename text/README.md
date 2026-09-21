# Text Analysis for Greek Papyri

## Purpose

This directory contains the text-processing part of the Greek papyri pipeline. Its job is to turn raw annotation transcripts into normalized Greek text, recover approximate word boundaries, and produce analyzable textual features for downstream study.

The module is designed for annotation files where Greek text is stored in a proxy Latin-letter transcription, often without reliable spacing and with damage markers. It keeps the original annotation-derived text traceable while creating cleaner text artifacts that can be compared with OCR and visual results elsewhere in the project.

## What This Module Does

The text module:

- reads annotation files associated with valid inscription/image pairs;
- extracts the `# Transcript` section from each `*_letters.txt` file;
- converts proxy characters such as `A`, `B`, `G`, `Q`, and `W` into Greek uppercase letters;
- normalizes Greek text by removing combining marks and uppercasing consistently;
- builds a Greek vocabulary from a Hugging Face inscription dataset when available, plus local seed words;
- segments continuous Greek character runs into likely word tokens;
- preserves damaged or uncertain text markers during segmentation;
- deduplicates repeated segmented transcripts;
- computes vocabulary frequency summaries;
- detects likely personal names using a local stem lexicon;
- clusters inscriptions using word and character TF-IDF features;
- extracts repeated cluster phrases and LDA topic summaries;
- prepares short, clean candidate texts for optional Aeneas dating;
- optionally runs a local Aeneas model if the required repository and model files are already present.

## How It Works

`main.py` is the orchestrator. It resolves the shared project layout, finds valid annotation/image pairs, loads transcripts, builds segmentation resources, runs analysis, and writes CSV outputs under the text output directory.

The first processing stage is transcription conversion. `transcription.py` reads each annotation file, extracts the transcript block, detects the usable encoding, converts the proxy alphabet into Greek letters, and normalizes the result. The conversion is intentionally explicit: only known proxy symbols are mapped, while whitespace and supported markers are preserved.

The second stage is segmentation. `segmentation.py` builds a frequency vocabulary from `Ericu950/Inscriptions_1` unless `--skip-hf-vocab` is used. It adds local seed and domain words with strong weights, then uses a dynamic-programming segmenter to choose likely token boundaries. The segmenter includes penalties for unknown words, very short unknown tokens, insertions, and wildcard damage markers.

The third stage is analysis. `analysis.py` filters clean Greek tokens, computes vocabulary statistics, matches possible personal names against a small stem lexicon, and builds document-level clustering features. Clustering combines word TF-IDF and character n-gram TF-IDF, chooses a cluster count with silhouette scoring, projects documents with Truncated SVD, extracts repeated phrases, and fits an LDA topic model.

The final optional stage prepares text for Aeneas. Candidate texts are cleaned, trimmed, scored by readability and damage ratio, and saved. If `--run-aeneas` is passed, the code expects a local `predictingthepast` repository and local Aeneas model files; it does not download them.

## Files in This Directory

| File | Role |
| --- | --- |
| `main.py` | Lightweight command-line entry point and orchestrator for the text pipeline. |
| `transcription.py` | Annotation reading, transcript extraction, proxy-to-Greek conversion, and Greek normalization. |
| `segmentation.py` | Vocabulary construction and dynamic-programming word segmentation. |
| `analysis.py` | Vocabulary statistics, name detection, clustering, topic modeling, Aeneas candidate preparation, and optional dating. |
| `__init__.py` | Marks the directory as an importable Python package. |
| `README.md` | Documentation for the text module. |

## Methods and Technologies

This module uses standard Python text processing together with data-science tools:

- `pandas` for tabular inputs and CSV outputs;
- `re`, `unicodedata`, `Counter`, and cached dynamic programming for normalization and segmentation;
- Hugging Face `datasets` as an optional source of inscription vocabulary;
- local weighted seed vocabularies for papyrological and Greek function words;
- scikit-learn TF-IDF vectorizers for word and character features;
- `KMeans` and cosine silhouette scoring for cluster selection;
- `TruncatedSVD` for low-dimensional cluster coordinates;
- `LatentDirichletAllocation` for topic summaries;
- optional JAX and `predictingthepast` imports for local Aeneas dating.

The approach is heuristic and corpus-informed rather than a fully supervised Greek NLP model. That is useful for this project because the transcriptions can be short, damaged, unsegmented, and inconsistent.

## Inputs and Outputs

The expected input is the shared project data layout used by the root pipeline. The text module mainly consumes annotation files, but `main.py` filters them through valid inscription/image pairs so that the text outputs stay aligned with the rest of the project.

Annotation files are expected to be named like:

```text
<inscription_id>_letters.txt
```

Inside each annotation file, the preferred transcript block is:

```text
# Transcript
ABG...
```

If a `# Transcript` header is missing, the loader falls back to the whole file text and emits a warning.

Important command-line options include:

- `--data-dir` for the shared data root;
- `--annotations-dir` for annotation files;
- `--image-dirs` for image directories used when finding valid pairs;
- `--output-dir` for generated artifacts;
- `--max-vocab-rows` to limit Hugging Face vocabulary loading;
- `--skip-hf-vocab` to use only local seed vocabulary;
- `--run-aeneas` to enable optional dating;
- `--aeneas-repo-dir` and `--aeneas-model-dir` for local Aeneas dependencies.

The module writes its outputs under the text output directory, normally `outputs/text/`:

| Output | Contents |
| --- | --- |
| `annotations_converted.csv` | Original filenames, inscription IDs, proxy text, converted Greek text, and detected encoding. |
| `segmented_transcripts.csv` | Deduplicated segmented token sequences per inscription. |
| `word_frequencies.csv` | Frequency table for clean Greek tokens. |
| `vocabulary_summary.csv` | Aggregate token count, vocabulary size, type-token ratio, hapax count, and rare-word count. |
| `detected_names.csv` | Token-level possible name matches. |
| `name_frequencies.csv` | Name lemma frequencies, observed forms, and inscription counts. |
| `name_gender_summary.csv` | Aggregate counts by inferred name gender. |
| `text_clusters.csv` | Document-level cluster assignments, text metrics, and SVD coordinates. |
| `text_cluster_silhouette.csv` | Candidate cluster counts and silhouette scores. |
| `cluster_repeated_phrases.csv` | High-scoring repeated phrases per cluster. |
| `text_topics_lda.csv` | LDA topic IDs and representative words. |
| `aeneas_candidate_texts.csv` | Cleaned and ranked candidate texts for Aeneas. |
| `aeneas_predictions.csv` | Optional Aeneas dating output when local model execution is enabled. |

## Role in the Full Project

The text module provides the linguistic side of the project. Vision and OCR modules work from images and predicted characters; this module works from annotation-derived transcripts. Its outputs make it possible to inspect vocabulary, repeated formulas, name patterns, thematic clusters, and possible chronological signals using text that has been normalized into a consistent Greek representation.

Because the module writes intermediate CSV files, its results are also useful for auditing. A developer can compare proxy text, converted Greek text, segmented tokens, and downstream analyses without rerunning the whole project.

## Notes for Future Development

Potential improvements include:

- adding evaluation data for segmentation quality;
- expanding the Greek name lexicon and separating uncertain matches from stronger matches;
- caching or versioning the external vocabulary source for reproducible runs;
- adding more explicit handling for damaged, bracketed, or restored letters;
- comparing segmented annotation text directly against OCR output;
- recording pipeline parameters in a run manifest beside the CSV outputs;
- replacing some heuristics with trained models if enough labeled papyrological text becomes available.

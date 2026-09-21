"""Image collection and preprocessing for Greek squeeze images."""

from __future__ import annotations

from pipeline_config import DatasetLayout, collect_image_records, find_valid_pairs


B1_TARGET_SIZE = (512, 512)


def collect_matched_images(layout: DatasetLayout, max_images: int | None = None):
    import pandas as pd

    valid_pairs = find_valid_pairs(layout)
    records = collect_image_records(layout.image_dirs)
    images_df = pd.DataFrame(records)
    if images_df.empty:
        return images_df
    images_df = images_df[images_df["inscription_id"].isin(valid_pairs)].copy()
    images_df = images_df.drop_duplicates("inscription_id").sort_values("inscription_id")
    if max_images is not None:
        images_df = images_df.head(max_images)
    return images_df.reset_index(drop=True)


def read_image_rgb(path):
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_and_pad(image, target_size=B1_TARGET_SIZE, pad_value=None):
    import cv2
    import numpy as np

    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if pad_value is None:
        pad_value = int(np.median(image))

    canvas = np.full((target_h, target_w), pad_value, dtype=resized.dtype)
    y_start = (target_h - new_h) // 2
    x_start = (target_w - new_w) // 2
    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized
    return canvas


def preprocess_image(path, target_size=B1_TARGET_SIZE):
    import cv2

    original_rgb = read_image_rgb(path)
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    standardized_gray = resize_and_pad(gray, target_size=target_size)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(standardized_gray)
    cleaned = cv2.fastNlMeansDenoising(contrast_enhanced, None, 7, 7, 21)
    return {
        "Original": original_rgb,
        "Standardized grayscale": standardized_gray,
        "Contrast enhanced": contrast_enhanced,
        "Final cleaned": cleaned,
    }


def preprocess_all_images(images_df):
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    cleaned_images = []
    records = []
    for idx, row in tqdm(
        images_df.reset_index(drop=True).iterrows(),
        total=len(images_df),
        desc="Preprocessing images",
    ):
        steps = preprocess_image(row["image_path"])
        cleaned = steps["Final cleaned"]
        cleaned_images.append(cleaned)
        records.append(
            {
                "array_index": idx,
                "inscription_id": row["inscription_id"],
                "filename": row["filename"],
                "image_folder": row["image_folder"],
                "image_path": str(row["image_path"]),
                "original_height": steps["Original"].shape[0],
                "original_width": steps["Original"].shape[1],
                "processed_height": cleaned.shape[0],
                "processed_width": cleaned.shape[1],
            }
        )

    if not cleaned_images:
        return np.empty((0, *B1_TARGET_SIZE), dtype=np.uint8), pd.DataFrame(records)
    return np.stack(cleaned_images).astype(np.uint8), pd.DataFrame(records)


def make_preprocessing_checks(preprocessed_df, cleaned_images):
    import pandas as pd

    return pd.DataFrame(
        {
            "inscription_id": preprocessed_df["inscription_id"],
            "mean_intensity": cleaned_images.mean(axis=(1, 2)),
            "std_intensity": cleaned_images.std(axis=(1, 2)),
            "min_intensity": cleaned_images.min(axis=(1, 2)),
            "max_intensity": cleaned_images.max(axis=(1, 2)),
        }
    )


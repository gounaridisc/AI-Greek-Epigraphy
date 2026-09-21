"""Line-crop preparation and OCR evaluation file helpers."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

from pipeline_config import DatasetLayout, collect_image_records, find_valid_pairs


B3_CROP_PADDING = 6


def collect_ocr_images(layout: DatasetLayout, max_images: int | None = None):
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


def rotation_matrix(image_shape, angle_rad):
    import cv2

    h, w = image_shape[:2]
    center = (w / 2.0, h / 2.0)
    return cv2.getRotationMatrix2D(center, math.degrees(angle_rad), 1.0)


def apply_matrix(points, matrix):
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return homogeneous @ matrix.T


def extract_lines(image_path, annotation_path):
    import cv2
    import numpy as np

    from ocr.evaluation import getRowBoxes, getRowTranscript, orderBoxes, readBoxFile

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    gangle, boxes, _, lines = readBoxFile(str(annotation_path))
    if not boxes:
        return []

    boxes2 = orderBoxes(boxes, gangle, lines)
    _, idlist = getRowBoxes(boxes2, gangle=gangle)
    gt_lines = getRowTranscript(str(annotation_path)).split("\n")
    matrix = rotation_matrix(gray.shape, gangle if gangle is not None else 0.0)

    h, w = gray.shape[:2]
    deskewed = cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderValue=int(np.median(gray)),
    )

    results = []
    for row_index, rowid in enumerate(idlist):
        if row_index >= len(gt_lines):
            break
        text = gt_lines[row_index].strip()
        if text == "":
            continue

        corners = []
        for index in rowid:
            box = boxes2[index]
            corners.extend(
                [
                    (box.xnw, box.ynw),
                    (box.xsw, box.ysw),
                    (box.xse, box.yse),
                    (box.xne, box.yne),
                ]
            )

        moved = apply_matrix(corners, matrix)
        x_min = int(max(0, np.floor(moved[:, 0].min()) - B3_CROP_PADDING))
        x_max = int(min(w, np.ceil(moved[:, 0].max()) + B3_CROP_PADDING))
        y_min = int(max(0, np.floor(moved[:, 1].min()) - B3_CROP_PADDING))
        y_max = int(min(h, np.ceil(moved[:, 1].max()) + B3_CROP_PADDING))
        if x_max - x_min < 5 or y_max - y_min < 5:
            continue
        results.append((deskewed[y_min:y_max, x_min:x_max], text))
    return results


def prepare_line_crops(images_df, annotations_dir):
    import pandas as pd
    from tqdm.auto import tqdm

    line_crops = []
    records = []
    skipped = 0

    for _, row in tqdm(images_df.iterrows(), total=len(images_df), desc="Cutting line crops"):
        annotation_path = Path(annotations_dir) / f"{row['inscription_id']}_letters.txt"
        if not annotation_path.exists():
            skipped += 1
            continue
        try:
            lines_out = extract_lines(row["image_path"], annotation_path)
        except Exception as error:
            skipped += 1
            print(f"Skipping {row['inscription_id']}: {error}")
            continue

        base_id = re.sub(r"_Rotation\d+_\d+dpi$", "", row["inscription_id"])
        for row_index, (crop, text) in enumerate(lines_out):
            records.append(
                {
                    "line_index": len(line_crops),
                    "inscription_id": row["inscription_id"],
                    "base_id": base_id,
                    "row_index": row_index,
                    "text": text,
                    "n_chars": len(text),
                }
            )
            line_crops.append(crop)

    lines_df = pd.DataFrame(records)
    print(f"Inscriptions processed: {len(images_df) - skipped}")
    print(f"Inscriptions skipped:   {skipped}")
    print(f"Total line crops:       {len(line_crops)}")
    return line_crops, lines_df


def split_line_crops(lines_df):
    import numpy as np
    from sklearn.model_selection import train_test_split

    if lines_df.empty:
        return lines_df

    unique_bases = lines_df["base_id"].unique()
    if len(unique_bases) < 2:
        lines_df = lines_df.copy()
        lines_df["split"] = "test"
        return lines_df

    train_bases, _ = train_test_split(unique_bases, test_size=0.2, random_state=42)
    train_bases = set(train_bases)
    lines_df = lines_df.copy()
    lines_df["split"] = np.where(lines_df["base_id"].isin(train_bases), "train", "test")
    return lines_df


def prepare_gt_dir(lines_df, annotations_dir, eval_root):
    gt_dir = Path(eval_root) / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for path in gt_dir.iterdir():
        if path.is_file():
            path.unlink()

    copied = 0
    test_ids = sorted(lines_df.loc[lines_df["split"] == "test", "inscription_id"].unique())
    for inscription_id in test_ids:
        if not inscription_id.endswith("_Rotation1_300dpi"):
            continue
        src = Path(annotations_dir) / f"{inscription_id}_letters.txt"
        if src.exists():
            shutil.copy(str(src), gt_dir / src.name)
            copied += 1
    return gt_dir, copied


def write_predictions_and_score(lines_df, pred_by_line, pred_dir_name, eval_root, gt_dir):
    from ocr.evaluation import run_evaluations

    pred_dir = Path(eval_root) / pred_dir_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    for path in pred_dir.iterdir():
        if path.is_file():
            path.unlink()

    test_rows = lines_df[lines_df["split"] == "test"].copy()
    for inscription_id, group in test_rows.groupby("inscription_id"):
        if not inscription_id.endswith("_Rotation1_300dpi"):
            continue
        group = group.sort_values("row_index")
        lines_out = [pred_by_line.get(line_index, "") for line_index in group["line_index"]]
        squeeze = inscription_id[: -len("_Rotation1_300dpi")]
        out_path = pred_dir / f"{squeeze}_transcript.txt"
        out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

    return run_evaluations(pred_dir=str(pred_dir), gt_dir=str(gt_dir))


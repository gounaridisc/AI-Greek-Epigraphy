"""Shared dataset path helpers for the Greek papyri pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class DatasetLayout:
    data_dir: Path
    annotations_dir: Path
    image_dirs: list[Path]
    output_dir: Path


def split_path_list(value: str | None) -> list[Path] | None:
    if not value:
        return None
    return [Path(part.strip()).expanduser() for part in value.split(",") if part.strip()]


def resolve_annotations_dir(data_dir: Path, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()

    candidates = [
        data_dir / "Annotations" / "Annotations",
        data_dir / "Annotations",
        data_dir / "annotations",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def resolve_image_dirs(data_dir: Path, override: str | None = None) -> list[Path]:
    override_dirs = split_path_list(override)
    if override_dirs:
        return [path.resolve() for path in override_dirs]

    numbered = [data_dir / f"Images{i}" for i in range(1, 5)]
    existing_numbered = [path.resolve() for path in numbered if path.exists()]
    if existing_numbered:
        return existing_numbered

    discovered = sorted(
        path.resolve()
        for path in data_dir.glob("Images*")
        if path.is_dir()
    )
    if discovered:
        return discovered

    fallback = data_dir / "images"
    return [fallback.resolve()]


def build_layout(args) -> DatasetLayout:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    return DatasetLayout(
        data_dir=data_dir,
        annotations_dir=resolve_annotations_dir(data_dir, args.annotations_dir),
        image_dirs=resolve_image_dirs(data_dir, args.image_dirs),
        output_dir=output_dir,
    )


def annotation_id_from_path(path: Path) -> str:
    name = path.name
    suffix = "_letters.txt"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def collect_annotation_ids(annotations_dir: Path) -> set[str]:
    if not annotations_dir.exists():
        return set()
    return {
        annotation_id_from_path(path)
        for path in annotations_dir.glob("*_letters.txt")
        if path.is_file()
    }


def collect_image_records(image_dirs: Iterable[Path]) -> list[dict]:
    records = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "inscription_id": path.stem,
                        "image_path": path,
                        "filename": path.name,
                        "image_folder": image_dir.name,
                    }
                )
    return records


def collect_image_ids(image_dirs: Iterable[Path]) -> set[str]:
    return {record["inscription_id"] for record in collect_image_records(image_dirs)}


def find_valid_pairs(layout: DatasetLayout) -> set[str]:
    return collect_annotation_ids(layout.annotations_dir) & collect_image_ids(layout.image_dirs)


def ensure_output_subdir(layout: DatasetLayout, name: str) -> Path:
    output_dir = layout.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def require_dataset(layout: DatasetLayout) -> None:
    missing = []
    if not layout.annotations_dir.exists():
        missing.append(f"annotations directory not found: {layout.annotations_dir}")
    existing_image_dirs = [path for path in layout.image_dirs if path.exists()]
    if not existing_image_dirs:
        missing.append(
            "no image directories found: "
            + ", ".join(str(path) for path in layout.image_dirs)
        )
    if missing:
        raise FileNotFoundError(
            "Dataset files are missing. See data/README.md.\n" + "\n".join(missing)
        )


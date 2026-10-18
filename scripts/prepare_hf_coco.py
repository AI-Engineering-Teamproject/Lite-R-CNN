"""Download and unpack the COCO 2017 files from the configured Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from huggingface_hub import hf_hub_download


REPO_ID = "Hyeon1222/COCO"
FILES = {
    "train": "train2017.zip",
    "val": "val2017.zip",
    "annotations": "annotations.zip",
}


def safe_members(archive: zipfile.ZipFile, prefix: str) -> list[zipfile.ZipInfo]:
    """Select files from the expected archive folder and reject unsafe paths."""
    selected = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        windows_path = PureWindowsPath(member.filename)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in path.parts
            or ".." in windows_path.parts
            or "\\" in member.filename
        ):
            raise ValueError(f"Unsafe path in {archive.filename}: {member.filename}")
        if not member.is_dir() and path.parts and path.parts[0] == prefix:
            selected.append(member)
    if not selected:
        raise ValueError(f"{archive.filename} contains no files under {prefix}/")
    return selected


def extract_split(zip_path: Path, destination: Path, prefix: str) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        members = safe_members(archive, prefix)
        extracted = 0
        for member in members:
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            # A previous run may have been interrupted after extracting only
            # part of a split. Keep complete files and repair missing/truncated
            # ones instead of treating any existing JPG as a complete dataset.
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            if target.stat().st_size != member.file_size:
                raise IOError(f"Extracted file size does not match ZIP entry: {target}")
            extracted += 1
    return extracted


def extract_annotations(zip_path: Path, destination: Path) -> list[str]:
    wanted = {
        "instances_train2017.json",
        "instances_val2017.json",
    }
    found = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            windows_path = PureWindowsPath(member.filename)
            if (
                path.is_absolute()
                or windows_path.is_absolute()
                or windows_path.drive
                or ".." in path.parts
                or ".." in windows_path.parts
                or "\\" in member.filename
            ):
                raise ValueError(f"Unsafe path in {archive.filename}: {member.filename}")
            if member.is_dir() or path.name not in wanted:
                continue
            target = destination / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            found.add(path.name)
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Annotation archive is missing: {', '.join(missing)}")
    return sorted(found)


def validate_annotation(path: Path) -> None:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not data.get("images") or not data.get("annotations") or not data.get("categories"):
        raise ValueError(f"{path} does not look like a populated COCO instances annotation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data/coco", help="COCO output directory (default: data/coco)"
    )
    parser.add_argument(
        "--repo-id", default=REPO_ID, help="Hugging Face dataset repo (default: %(default)s)"
    )
    parser.add_argument(
        "--cache-dir",
        default="data/hf_cache",
        help="Hugging Face download cache; downloaded ZIPs are retained here",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=(*FILES.keys(), "all"),
        default=["all"],
        help="Files to download and extract (default: all)",
    )
    args = parser.parse_args()
    requested = set(FILES if "all" in args.splits else args.splits)
    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    image_dir, annotation_dir = data_dir / "images", data_dir / "annotations"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "annotations"):
        if split not in requested:
            continue
        filename = FILES[split]
        print(f"Downloading {args.repo_id}/{filename} via Hugging Face Hub...")
        zip_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=cache_dir,
            )
        )
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Downloaded file is not a ZIP archive: {zip_path}")
        if split in ("train", "val"):
            prefix = "train2017" if split == "train" else "val2017"
            expected_dir = image_dir / prefix
            count = extract_split(zip_path, image_dir, prefix)
            if count:
                print(f"Extracted or repaired {count} files in {expected_dir}")
            else:
                print(f"All files in {expected_dir} already match the ZIP archive")
        else:
            extract_annotations(zip_path, annotation_dir)
            print(f"Extracted instance annotations to {annotation_dir}")

    requirements = {
        "train": image_dir / "train2017",
        "val": image_dir / "val2017",
        "annotations": annotation_dir,
    }
    if "train" in requested and not any(requirements["train"].glob("*.jpg")):
        raise FileNotFoundError(f"No training images found in {requirements['train']}")
    if "val" in requested and not any(requirements["val"].glob("*.jpg")):
        raise FileNotFoundError(f"No validation images found in {requirements['val']}")
    for split in ("train", "val"):
        annotation = annotation_dir / f"instances_{split}2017.json"
        if "annotations" in requested:
            validate_annotation(annotation)
    print(f"COCO data is ready at {data_dir.resolve()}")


if __name__ == "__main__":
    main()

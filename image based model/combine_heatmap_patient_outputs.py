"""Combine native PNG outputs from Heatmap_patient.py into one comparison figure.

This script deliberately does not parse clinical source files or redraw clinical
data. It only crops whitespace, rescales, and arranges PNG files produced by the
patient plotter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


CATEGORIES = ("asm", "seizure", "eeg")
CATEGORY_LABELS = {
    "asm": "Antiseizure medication",
    "seizure": "Seizure",
    "eeg": "EEG",
}

COLUMN_WIDTH = 2200
PAGE_MARGIN = 160
COLUMN_GAP = 120
ROW_GAP = 120
HEADER_HEIGHT = 250


def content_crop(image: Image.Image, padding: int = 28) -> Image.Image:
    """Crop near-white outer margins while retaining a small border."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    difference = difference.point(lambda value: 0 if value < 8 else 255)
    bounding_box = difference.getbbox()

    if bounding_box is None:
        return rgb

    left, top, right, bottom = bounding_box
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def fit_width(image: Image.Image, width: int) -> Image.Image:
    """Resize proportionally to a fixed width."""
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def load_font(filename: str, size: int) -> ImageFont.ImageFont:
    """Load a Windows font, with a portable Pillow fallback."""
    font_path = Path("C:/Windows/Fonts") / filename
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 checksum for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_native(patient_dir: Path, category: str) -> Path:
    """Find the unique native heatmap PNG for a category."""
    matches = sorted(patient_dir.glob(f"*_{category}_heatmap.png"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one '*_{category}_heatmap.png' in "
            f"'{patient_dir}', found {len(matches)}."
        )
    return matches[0]


def combine(
    root: Path,
    dre_dir_name: str,
    ndre_dir_name: str,
    output_stem: Path,
) -> tuple[Path, Path, Path]:
    """Compose the native DRE and NDRE panels and write PNG/PDF/manifest."""
    patient_dirs = [root / dre_dir_name, root / ndre_dir_name]
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            raise FileNotFoundError(
                f"Native plot directory does not exist: '{patient_dir}'. "
                "Run Heatmap_patient.py first."
            )

    inputs: list[dict[str, str]] = []
    rows: list[tuple[Image.Image, Image.Image]] = []

    for category in CATEGORIES:
        row_images: list[Image.Image] = []
        for patient_dir in patient_dirs:
            path = find_native(patient_dir, category)
            inputs.append(
                {
                    "category": category,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
            )
            with Image.open(path) as source:
                row_images.append(fit_width(content_crop(source), COLUMN_WIDTH))
        rows.append((row_images[0], row_images[1]))

    row_heights = [max(left.height, right.height) for left, right in rows]
    page_width = PAGE_MARGIN * 2 + COLUMN_WIDTH * 2 + COLUMN_GAP
    page_height = (
        PAGE_MARGIN
        + HEADER_HEIGHT
        + sum(row_heights)
        + ROW_GAP * (len(rows) - 1)
        + PAGE_MARGIN
    )

    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font("arialbd.ttf", 86)
    column_font = load_font("arialbd.ttf", 62)
    panel_font = load_font("arialbd.ttf", 58)

    draw.text(
        (PAGE_MARGIN, 52),
        "Twelve-month clinical comparison of representative DRE and NDRE cases",
        fill="#111111",
        font=title_font,
    )

    left_x = PAGE_MARGIN
    right_x = PAGE_MARGIN + COLUMN_WIDTH + COLUMN_GAP
    header_y = PAGE_MARGIN + 65
    draw.text(
        (left_x + COLUMN_WIDTH / 2, header_y),
        "DRE case",
        fill="#B23A48",
        font=column_font,
        anchor="mm",
    )
    draw.text(
        (right_x + COLUMN_WIDTH / 2, header_y),
        "NDRE case",
        fill="#256D85",
        font=column_font,
        anchor="mm",
    )

    y = PAGE_MARGIN + HEADER_HEIGHT
    for index, (left, right) in enumerate(rows):
        row_height = row_heights[index]
        page.paste(left, (left_x, y))
        page.paste(right, (right_x, y))
        draw.text(
            (46, y + 20),
            chr(ord("A") + index),
            fill="#111111",
            font=panel_font,
        )
        y += row_height + ROW_GAP

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    manifest_path = output_stem.with_name(
        f"{output_stem.name}_manifest"
    ).with_suffix(".json")

    page.save(png_path, dpi=(600, 600), optimize=True)
    page.save(pdf_path, "PDF", resolution=600.0)

    manifest = {
        "purpose": "Conference comparison figure assembled from native plotter PNGs",
        "plotter": str(Path("Heatmap_patient.py").resolve()),
        "plotter_command": (
            '.\\.venv\\Scripts\\python.exe Heatmap_patient.py '
            '"DRE\\0208 DRE5.docx" "NDRE\\0085 NS1.docx" '
            '--months-to-plot 12 '
            '--output-dir "heatmap_patient_native_comparison"'
        ),
        "composition_note": (
            "No clinical data were parsed or replotted. The compositor only "
            "cropped whitespace, scaled, and arranged native Heatmap_patient.py PNGs."
        ),
        "included_categories": [
            {"panel": chr(ord("A") + index), "key": category, "label": CATEGORY_LABELS[category]}
            for index, category in enumerate(CATEGORIES)
        ],
        "inputs": inputs,
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "width_px": page.width,
            "height_px": page.height,
            "dpi": 600,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return png_path, pdf_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine native Heatmap_patient.py PNGs into a DRE/NDRE figure."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("heatmap_patient_native_comparison"),
        help="Directory containing one native-output subfolder per patient.",
    )
    parser.add_argument("--dre-dir", default="0003 DRE3")
    parser.add_argument("--ndre-dir", default="0031 NS1")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(
            "heatmap_patient_native_comparison/heatmap_patient_native_comparison"
        ),
        help="Output path without an extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = combine(args.root, args.dre_dir, args.ndre_dir, args.output_stem)
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()

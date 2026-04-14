from pathlib import Path

import json
import io
from typing import Any

from datasets import load_dataset
from PIL import Image

out_root = Path("dataset/mathvision")

img_dir = out_root / "images"

out_root.mkdir(parents=True, exist_ok=True)

img_dir.mkdir(parents=True, exist_ok=True)

split_name = "testmini"  # or "test"


def _load_mathvision_split(split: str):

    try:
        # Preferred path: canonical dataset loader.
        return load_dataset("MathLLMs/MathVision", split=split)
    except TypeError as e:
        # Compatibility fallback for older `datasets` versions that fail while
        # parsing feature metadata from cache.
        msg = str(e)
        if "dataclass type or instance" not in msg:
            raise

        try:
            return load_dataset(
                "MathLLMs/MathVision",
                split=split,
                revision="refs/convert/parquet",
            )
        except Exception as e2:
            raise RuntimeError(
                "Failed to load MathVision with both default loader and parquet fallback. "
                "Please upgrade `datasets` (recommended >= 2.20) and retry.") from e2


ds = _load_mathvision_split(split_name)

rows = []


def _extract_extension_from_path(path_value: Any) -> str | None:

    if path_value is None:
        return None

    ext = Path(str(path_value)).suffix.lower()
    return ext if ext else None


def _jsonable(value: Any) -> Any:

    if value is None or isinstance(value, (str, int, float, bool)):

        return value

    if isinstance(value, Path):

        return str(value)

    if isinstance(value, dict):

        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):

        return [_jsonable(v) for v in value]

    # Fallback for HF/PIL/custom objects that are not directly JSON
    # serializable.

    return str(value)


def _extract_pil_image(
        ex: dict[str, Any]) -> tuple[Image.Image | None, str | None]:

    ext: str | None = None

    decoded = ex.get("decoded_image")
    pil_image: Image.Image | None = None
    if isinstance(decoded, Image.Image):
        pil_image = decoded

    image_value = ex.get("image")
    if isinstance(image_value, dict):
        ext = _extract_extension_from_path(image_value.get("path"))

    elif isinstance(image_value, (str, Path)):
        ext = _extract_extension_from_path(image_value)

    if isinstance(image_value, Image.Image) and pil_image is None:
        pil_image = image_value

    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        if image_bytes is not None and pil_image is None:
            pil_image = Image.open(io.BytesIO(image_bytes))

        image_path = image_value.get("path")
        if image_path and pil_image is None:
            pil_image = Image.open(image_path)

    elif isinstance(image_value, (str, Path)) and pil_image is None:
        pil_image = Image.open(image_value)

    if ext is None and pil_image is not None and pil_image.format:
        fmt = pil_image.format.lower()
        ext = ".jpg" if fmt == "jpeg" else f".{fmt}"

    if pil_image is None:
        return None, ext

    return pil_image.convert("RGB"), ext


for i, ex in enumerate(ds):

    qid = str(ex.get("id", i))

    pil_image, image_ext = _extract_pil_image(ex)

    if pil_image is None:

        # If no image can be extracted, keep the sample but leave image path
        # null.
        saved_rel_path = None

    else:

        image_ext = image_ext or ".png"
        img_path = img_dir / f"{qid}{image_ext}"

        pil_image.save(img_path)
        saved_rel_path = str(Path("images") / f"{qid}{image_ext}")

    record = {k: _jsonable(v) for k, v in ex.items()}
    # Keep all original keys and replace decoded image object with on-disk
    # path.
    record["decoded_image"] = saved_rel_path
    rows.append(record)

with (out_root / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:

    for r in rows:

        f.write(json.dumps(r, ensure_ascii=False) + "\n")

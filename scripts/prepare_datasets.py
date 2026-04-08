from pathlib import Path
import json
from datasets import load_dataset

out_root = Path("data/mathvision")
img_dir = out_root / "images"
out_root.mkdir(parents=True, exist_ok=True)
img_dir.mkdir(parents=True, exist_ok=True)

split_name = "testmini"  # or "test"
ds = load_dataset("MathLLMs/MathVision", split=split_name)

rows = []
for i, ex in enumerate(ds):
    qid = str(ex.get("id", i))
    img_path = img_dir / f"{qid}.png"

    # decoded_image is an HF Image feature (PIL Image)
    if ex.get("decoded_image") is not None:
        ex["decoded_image"].save(img_path)
    else:
        # fallback if only path string exists and you handle separately
        continue

    rows.append({
        "id": qid,
        "image": str(Path("images") / f"{qid}.png"),
        "question": ex["question"],
        "answer": ex["answer"],
    })

with (out_root / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
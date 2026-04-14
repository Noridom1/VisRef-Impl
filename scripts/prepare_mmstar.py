from datasets import load_dataset
from pathlib import Path
import json

# ====== CONFIG ======
root_dir = Path("./data/mmstar")
image_dir = root_dir / "images"
image_dir.mkdir(parents=True, exist_ok=True)

jsonl_path = root_dir / "testmini.jsonl"

# ====== LOAD DATASET ======
dataset = load_dataset("Lin-Chen/MMStar", split="val")

print(f"Dataset size: {len(dataset)}")
print("Sample:", dataset[0])

# ====== PROCESS ======


def format_options(choices):
    """
    Convert list of choices to:
    A: xxx, B: xxx, ...
    """
    option_letters = ["A", "B", "C", "D", "E"]
    formatted = []
    for i, choice in enumerate(choices):
        formatted.append(f"{option_letters[i]}: {choice}")
    return ", ".join(formatted)


with open(jsonl_path, "w", encoding="utf-8") as f:
    for idx, item in enumerate(dataset):
        # ---- Save image ----
        img = item["image"]
        img_path = image_dir / f"{idx}.jpg"
        img.save(img_path)

        # ---- Handle choices ----
        choices = item.get("choices", [])
        options_text = format_options(choices)

        # ---- Convert answer ----
        answer = item.get("answer")

        # If answer is index (e.g., 0 → A)
        if isinstance(answer, int):
            answer_letter = chr(ord("A") + answer)
        else:
            answer_letter = answer  # already A/B/C...

        # ---- Build question ----
        question = item["question"]
        if options_text:
            question = f"{question}\nOptions: {options_text}"

        # ---- Final record ----
        record = {
            "index": idx,
            "question": question,
            "image": img,
            "answer": answer_letter,
            "category": "coarse perception",  # or map if MMStar has category
            "l2_category": "image scene and topic",
            "meta_info": {
                "source": "MMStar",
                "split": "val",
                "image_path": str(img_path),
            },
        }

        # IMPORTANT: usually you want path, not PIL object
        record["image"] = str(img_path)

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved to {jsonl_path}")

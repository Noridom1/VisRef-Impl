from datasets import load_dataset

import os

import json

from tqdm import tqdm


# ====== CONFIG ======

root_dir = "./data/mathvista"

image_dir = os.path.join(root_dir, "images")

os.makedirs(image_dir, exist_ok=True)


jsonl_path = os.path.join(root_dir, "testmini.jsonl")


# ====== LOAD DATASET ======

dataset = load_dataset("AI4Math/MathVista", split="testmini")


print(f"Dataset size: {len(dataset)}")


# ====== PROCESS & SAVE ======

with open(jsonl_path, "w", encoding="utf-8") as f:

    for sample in tqdm(dataset):

        try:

            pid = sample["pid"]  # use original id
            image_filename = f"{pid}.png"
            image_path = os.path.join(image_dir, image_filename)

            # ====== SAVE IMAGE ======

            image = sample.get("decoded_image", None)

            if image is None:

                image = sample["image"]  # fallback (should still be PIL)

            image.save(image_path)

            # ====== BUILD JSON LINE ======

            record = {
                "pid": pid,
                "image": f"images/{image_filename}",
                "question": sample["question"],
                "answer": sample.get("answer"),
                "choices": sample.get("choices"),
                "unit": sample.get("unit"),
                "precision": sample.get("precision"),
                "question_type": sample.get("question_type"),
                "answer_type": sample.get("answer_type"),
                "metadata": sample.get("metadata"),
                "query": sample.get("query"),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:

            print(f"Error at pid {sample.get('pid')}: {e}")


print("✅ Done! Dataset exported to ./data/mathvista/")

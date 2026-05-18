import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# =========================
# User inputs
# =========================
image_path = "data/images/4.jpg"
user_prompt = "How many different digits can you find in this picture?"

# =========================
# Load model + processor
# =========================
model_name = "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # use float16 if your GPU doesn't support bf16
    device_map="auto",
    # attn_implementation="flash_attention_2",  # optional, faster if installed
)

processor = AutoProcessor.from_pretrained(model_name)

# =========================
# Load image
# =========================
image = Image.open(image_path).convert("RGB")

# =========================
# Build chat messages
# =========================
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant for answering questions about images. Think step by step.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": user_prompt,
            },
        ],
    }
]

# =========================
# Prepare inputs
# =========================
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

# =========================
# Generate
# =========================
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
)

# Remove prompt tokens
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n=== Model Output ===")
print(output_text[0])
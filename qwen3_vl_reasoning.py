"""Run Qwen3-VL on a single CS photo and print a rationale."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_DATA_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/data/CS_Xplots_2019_2023_test"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_first_image(data_dir: Path) -> Path:
    """Return the first image file found under data_dir (recursive)."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            return path
    raise FileNotFoundError(f"No images found under: {data_dir}")


def load_image(image_path: Path) -> Image.Image:
    """Load an image as RGB for model input."""
    with Image.open(image_path) as img:
        return img.convert("RGB")


def build_prompt() -> str:
    """Return the user prompt for the rationale task."""
    return (
        "Explain what you see in this CS (community science) photo. "
        "Provide a short rationale about the key visual cues."
    )


def load_qwen_and_processor(model_id: str, use_bfloat16: bool):
    """Load the HF processor and model. Adjust classes here if needed."""
    # NOTE: Qwen3-VL may require a specific model class from transformers.
    # If AutoModelForCausalLM fails, replace it with the recommended class from HF.

    dtype = torch.bfloat16 if use_bfloat16 else "auto"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="auto"
        )

    return model, processor


def prepare_inputs(processor, image_path: Path, prompt: str):
    """Prepare model inputs; adjust for chat templates if available."""
    if hasattr(processor, "apply_chat_template"):
        image_uri = str(image_path.resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        return formatted

    # Fallback: simple text+image encoding.
    image = load_image(image_path)
    return processor(text=prompt, images=image, return_tensors="pt")


def generate_rationale(model, processor, image_path: Path, prompt: str) -> str:
    """Generate a single rationale response."""
    inputs = prepare_inputs(processor, image_path, prompt)
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
    )
    output_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a CS photo with Qwen3-VL.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to an image. If omitted, uses the first image in the data dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory to search for images when --image is not provided.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 weights if supported by your hardware.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image or find_first_image(args.data_dir)
    prompt = build_prompt()

    model, processor = load_qwen_and_processor(args.model_id, args.bfloat16)
    rationale = generate_rationale(model, processor, image_path, prompt)

    print(f"Image: {image_path}")
    print("\nRationale:\n")
    print(rationale)


if __name__ == "__main__":
    main()

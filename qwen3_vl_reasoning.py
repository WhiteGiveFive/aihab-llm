"""Run batched Qwen3-VL rationales for CS photos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils import (
    add_image_paths,
    attach_rationales,
    get_habitat_attrs,
    load_samples,
    safe_model_id,
    write_rows_with_rationale,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_IMAGE_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/data/CS_Xplots_2019_2023_test"
)
DATA_TABLES_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-llm/data_tables"
)
OUTPUT_DIR = DATA_TABLES_DIR / "qwen3_vl_outputs"
def load_split_rows(csv_path: Path, split: str):
    rows = load_samples(csv_path, split=split)
    return add_image_paths(rows, base_dir=BASE_IMAGE_DIR)

def build_messages(rows):
    all_messages = []
    for row in rows:
        image_path = Path(row["image_path"]).resolve()
        label = row.get("ground_truth_word_label", "").strip()
        attrs = get_habitat_attrs(label) if label else None
        prompt = build_prompt(label, attrs)
        messages = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": "You are a helpful ecologist."}
                ],
            }, 
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        all_messages.append(messages)
    return all_messages

def batched_infer(model, processor, rows, batch_size=4):
    # Ensure left padding for batch generation (Qwen3-VL guidance)
    processor.tokenizer.padding_side = "left"

    all_messages = build_messages(rows)

    results = []
    total_batches = (len(all_messages) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(all_messages), batch_size),
        total=total_batches,
        desc="batches",
    ):
        batch_messages = all_messages[start : start + batch_size]

        # Build inputs for the batch
        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device)

        # Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )

        # Trim prompt tokens and decode
        output_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_texts = processor.batch_decode(
            output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results.extend(output_texts)

    return results


def print_sample_paths(image_paths, split, sample_size=3):
    if not image_paths:
        print(f"No image paths for {split}.")
        return
    count = min(sample_size, len(image_paths))
    print(f"Sample image paths for {split} (showing {count} of {len(image_paths)}):")
    for idx in range(count):
        print(f"  {idx + 1}. {image_paths[idx]}")


def print_sample_prompts(rows, split, sample_size=2):
    if not rows:
        print(f"No rows for {split} to sample prompts.")
        return
    count = min(sample_size, len(rows))
    print(f"Sample prompts for {split} (showing {count} of {len(rows)}):")
    for idx in range(count):
        row = rows[idx]
        label = row.get("ground_truth_word_label", "").strip()
        attrs = get_habitat_attrs(label) if label else None
        prompt = build_prompt(label, attrs)
        print(f"\n--- Prompt {idx + 1} ({row.get('file_name', 'unknown')}): ---")
        print(prompt)


def build_prompt(label: str, attrs: dict | None) -> str:
    """Return a prompt asking for a 1-5 consistency score with rationale."""
    header = (
        "You are given a ground-level habitat photo and its ground-truth label. "
        "Assess how consistent the photo is with the label."
    )
    score_instructions = (
        "Score the consistency from 1 to 5 "
        "(5 = strong match, 1 = poor match). "
        "Provide a short rationale based on visual cues."
    )
    label_line = f"Ground-truth class: {label}" if label else "Ground-truth class: (unknown)"
    if attrs:
        attr_lines = "\n".join(f"- {key}: {value}" for key, value in attrs.items())
        attr_block = f"Typical visual attributes:\n{attr_lines}"
    else:
        attr_block = "Typical visual attributes: (not available)"
    output_format = (
        "Respond ONLY with valid JSON in the exact format:\n"
        '{"score": <1-5>, "rationale": "<short text>"}'
    )
    return "\n\n".join([header, label_line, attr_block, score_instructions, output_format])


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched Qwen3-VL rationales for CS photos."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--sample-paths",
        type=int,
        default=0,
        help="Number of sample image paths to print per split.",
    )
    parser.add_argument(
        "--sample-prompts",
        type=int,
        default=0,
        help="Number of sample prompts to print per split.",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 weights if supported by your hardware.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, processor = load_qwen_and_processor(args.model_id, args.bfloat16)
    model_tag = safe_model_id(args.model_id)

    splits = [
        ("mis", DATA_TABLES_DIR / "mis.csv"),
        ("correct", DATA_TABLES_DIR / "correct.csv"),
    ]

    for split, csv_path in splits:
        rows = load_split_rows(csv_path, split=split)
        if not rows:
            print(f"No rows found for {split}; skipping.")
            continue

        if args.sample_paths > 0:
            image_paths = [row["image_path"] for row in rows]
            print_sample_paths(image_paths, split, sample_size=args.sample_paths)
        if args.sample_prompts > 0:
            print_sample_prompts(rows, split, sample_size=args.sample_prompts)

        rationales = batched_infer(
            model, processor, rows, batch_size=args.batch_size
        )
        updated_rows = attach_rationales(rows, rationales)
        out_path = OUTPUT_DIR / f"{split}_{model_tag}.csv"
        write_rows_with_rationale(updated_rows, out_path)
        print(f"Wrote {len(updated_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

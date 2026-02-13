"""Run Ministral 3 on habitat photos with score/top3/all-L3 prompt modes.

References:
- https://huggingface.co/collections/mistralai/ministral-3
- qwen3_vl_reasoning.py
- qwen3_vl_rationale_top3.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

from cs_hab import REASSIGN_LABEL_NAME_L3
from utils import (
    add_image_paths,
    attach_rationales,
    get_habitat_attrs,
    load_samples,
    safe_model_id,
    write_rows_with_rationale,
)

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
BASE_IMAGE_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/data/CS_Xplots_2019_2023_test"
)
DATA_TABLES_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-llm/data_tables"
)
OUTPUT_DIR = DATA_TABLES_DIR / "ministral3_outputs"


def load_split_rows(csv_path: Path, split: str, base_dir: Path):
    """Load a CSV split and attach image paths."""
    rows = load_samples(csv_path, split=split)
    return add_image_paths(rows, base_dir=base_dir)


def map_l3_id_to_name(l3_id: str) -> str:
    """Map an L3 numeric id (string in CSV) to name."""
    if l3_id is None:
        raise ValueError("L3 id is missing.")
    raw = str(l3_id).strip()
    if raw == "":
        raise ValueError("L3 id is empty.")
    try:
        idx = int(raw)
    except ValueError:
        raise ValueError(f"L3 id must be an integer value: {l3_id!r}")
    try:
        return REASSIGN_LABEL_NAME_L3[idx]
    except KeyError as exc:
        valid = sorted(REASSIGN_LABEL_NAME_L3.keys())
        raise ValueError(f"Unknown L3 id {idx}. Expected one of: {valid}") from exc


def build_prompt_top3(candidates: List[Dict[str, str]]) -> str:
    """Build forced-choice prompt for three candidate habitats."""
    if len(candidates) != 3:
        raise ValueError(f"Expected 3 candidates, got {len(candidates)}.")
    header = (
        "You are given a ground-level habitat photo and three candidate habitat classes with typical visual descriptions. "
        "Analyse the photo and choose exactly one habitat from the candidates that best matches the photo. "
        "Ignore non-habitat, human-made objects (e.g., people, bags, equipment, panels) and use only habitat cues."
    )
    payload = {"candidates": candidates}
    candidate_block = (
        "Candidates and their descriptions(JSON): "
        f"{json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}"
    )
    constraints = (
        "predicted habitat must be exactly one of the candidate names. "
        "Provide a short rationale based on visible cues."
    )
    output_format = (
        "Respond ONLY with valid JSON in the exact format:\n"
        '{"pred_candidate":"<one of candidates>","rationale":"<short text>"}'
    )
    return "\n\n".join([header, candidate_block, constraints, output_format])


def build_prompt_all_l3_names_only(l3_names: List[str]) -> str:
    """Build forced-choice prompt using all L3 habitat names (no attributes)."""
    header = (
        "You are given a ground-level habitat photo and a list of candidate habitat classes. "
        "Select exactly one habitat that best matches the photo. "
        "Ignore non-habitat, human-made objects (e.g., people, bags, equipment, panels) and use only habitat cues."
    )
    candidate_block = f"Candidates: {json.dumps(l3_names, ensure_ascii=True)}"
    constraints = (
        "predicted habitat must be exactly one of the candidate names. "
        "Provide a short rationale based on visible cues."
    )
    output_format = (
        "Respond ONLY with valid JSON in the exact format:\n"
        '{"pred_habitat":"<one of candidates>","rationale":"<short text>"}'
    )
    return "\n\n".join([header, candidate_block, constraints, output_format])


def build_prompt_score(label: str, attrs: dict | None) -> str:
    """Build score prompt asking for consistency 1-5 plus rationale."""
    header = (
        "You are given a ground-level habitat photo and its ground-truth label. "
        "Assess how consistent the photo is with the label. "
        "Ignore non-habitat, human-made objects (e.g., people, bags, equipment, panels) and use only habitat cues."
    )
    score_instructions = (
        "Score the consistency from 1 to 5 (5 = strong match, 1 = poor match). "
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


def _build_ministral_message(image_path: Path, prompt: str) -> List[Dict]:
    """Build a single-user Ministral chat message with local file URI."""
    # image_uri = image_path.resolve().as_uri()   # would have issues with some unconventional file names in the test set.
    resolved = image_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    image_uri = f"file://{resolved}"
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful ecologist."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_messages(rows: List[Dict[str, str]], prompt_mode: str):
    """Create per-row chat messages for Ministral 3."""

    def format_attrs(attrs: Optional[Dict[str, str]]) -> str:
        if not attrs:
            return "(not available)"
        return "; ".join(f"{key}: {value}" for key, value in attrs.items())

    all_l3_names = None
    if prompt_mode == "all_l3_names":
        all_l3_names = [name for _, name in sorted(REASSIGN_LABEL_NAME_L3.items())]

    all_messages = []
    for row in rows:
        image_path = Path(row["image_path"]).resolve()

        if prompt_mode == "score":
            label = row.get("ground_truth_word_label", "").strip()
            attrs = get_habitat_attrs(label) if label else None
            prompt = build_prompt_score(label, attrs)
        elif prompt_mode == "top3":
            ids = [row["top3_label_1"], row["top3_label_2"], row["top3_label_3"]]
            names = [map_l3_id_to_name(l3_id) for l3_id in ids]
            attr_dicts = [get_habitat_attrs(name) for name in names]
            candidates = [
                {"name": name, "attrs": format_attrs(attrs)}
                for name, attrs in zip(names, attr_dicts)
            ]
            prompt = build_prompt_top3(candidates)
        elif prompt_mode == "all_l3_names":
            prompt = build_prompt_all_l3_names_only(all_l3_names)
        else:
            raise ValueError(f"Unknown prompt mode: {prompt_mode}")

        all_messages.append(_build_ministral_message(image_path, prompt))
    return all_messages


def load_ministral_and_tokenizer(model_id: str, use_bfloat16: bool):
    """Load Ministral 3 model/tokenizer for multimodal generation."""
    tokenizer = MistralCommonBackend.from_pretrained(model_id)
    model_kwargs = {"device_map": "auto"}
    if use_bfloat16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = Mistral3ForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    return model, tokenizer


def batched_infer(
    model,
    tokenizer,
    rows,
    max_new_tokens: int,
    prompt_mode: str,
):
    """Run inference one sample at a time for Ministral 3."""
    all_messages = build_messages(rows, prompt_mode=prompt_mode)

    results = []
    for messages in tqdm(all_messages, total=len(all_messages), desc="samples"):
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        tokenized = tokenized.to(model.device)

        if "pixel_values" not in tokenized:
            raise ValueError("Tokenizer output missing pixel_values for multimodal input.")
        image_sizes = [tokenized["pixel_values"].shape[-2:]]

        output = model.generate(
            **tokenized,
            image_sizes=image_sizes,
            max_new_tokens=max_new_tokens,
        )[0]

        prompt_len = tokenized["input_ids"].shape[-1]   # for a single input, the input_ids is a tensor with shape [1, input_sequence_length]
        decoded = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip() # different from original but workable
        results.append(decoded)

    return results


def print_sample_paths(image_paths, split, sample_size=3):
    """Print a few image paths for quick inspection."""
    if not image_paths:
        print(f"No image paths for {split}.")
        return
    count = min(sample_size, len(image_paths))
    print(f"Sample image paths for {split} (showing {count} of {len(image_paths)}):")
    for idx in range(count):
        print(f"  {idx + 1}. {image_paths[idx]}")


def print_sample_prompts(rows, split, sample_size: int, prompt_mode: str):
    """Print a few constructed prompts for quick inspection."""
    if not rows:
        print(f"No rows for {split} to sample prompts.")
        return
    count = min(sample_size, len(rows))
    print(f"Sample prompts for {split} (showing {count} of {len(rows)}):")

    all_l3_names = None
    if prompt_mode == "all_l3_names":
        all_l3_names = [name for _, name in sorted(REASSIGN_LABEL_NAME_L3.items())]

    for idx in range(count):
        row = rows[idx]
        if prompt_mode == "score":
            label = row.get("ground_truth_word_label", "").strip()
            attrs = get_habitat_attrs(label) if label else None
            prompt = build_prompt_score(label, attrs)
        elif prompt_mode == "top3":
            ids = [row["top3_label_1"], row["top3_label_2"], row["top3_label_3"]]
            names = [map_l3_id_to_name(l3_id) for l3_id in ids]
            attr_dicts = [get_habitat_attrs(name) for name in names]
            candidates = [
                {
                    "name": name,
                    "attrs": "; ".join(
                        f"{key}: {value}" for key, value in (attrs or {}).items()
                    )
                    if attrs
                    else "(not available)",
                }
                for name, attrs in zip(names, attr_dicts)
            ]
            prompt = build_prompt_top3(candidates)
        else:
            prompt = build_prompt_all_l3_names_only(all_l3_names)
        print(f"\n--- Prompt {idx + 1} ({row.get('file_name', 'unknown')}): ---")
        print(prompt)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Ministral 3 rationales and habitat selection for CS photos."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="score",
        choices=("score", "top3", "all_l3_names"),
        help="Prompt style: score consistency, choose from top3, or choose from all L3 names.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per sample.",
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
    parser.add_argument(
        "--base-image-dir",
        type=Path,
        default=BASE_IMAGE_DIR,
        help="Base directory containing CS images.",
    )
    parser.add_argument(
        "--data-tables-dir",
        type=Path,
        default=DATA_TABLES_DIR,
        help="Directory containing mis/correct CSVs and output folder.",
    )
    return parser.parse_args()


def _output_suffix(prompt_mode: str) -> str:
    if prompt_mode == "top3":
        return "choose_from_top3"
    if prompt_mode == "all_l3_names":
        return "choose_from_all_l3"
    return ""


def main() -> None:
    args = parse_args()

    model, tokenizer = load_ministral_and_tokenizer(args.model_id, args.bfloat16)
    model_tag = safe_model_id(args.model_id)

    base_image_dir = args.base_image_dir
    data_tables_dir = args.data_tables_dir
    output_dir = data_tables_dir / "ministral3_outputs"

    splits = [
        ("mis", data_tables_dir / "mis.csv"),
        ("correct", data_tables_dir / "correct.csv"),
    ]

    for split, csv_path in splits:
        rows = load_split_rows(csv_path, split=split, base_dir=base_image_dir)
        if not rows:
            print(f"No rows found for {split}; skipping.")
            continue

        if args.sample_paths > 0:
            image_paths = [row["image_path"] for row in rows]
            print_sample_paths(image_paths, split, sample_size=args.sample_paths)
        if args.sample_prompts > 0:
            print_sample_prompts(
                rows,
                split,
                sample_size=args.sample_prompts,
                prompt_mode=args.prompt_mode,
            )

        outputs = batched_infer(
            model,
            tokenizer,
            rows,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
        )
        updated_rows = attach_rationales(rows, outputs)

        suffix = _output_suffix(args.prompt_mode)
        if suffix:
            out_path = output_dir / f"{split}_{model_tag}_{suffix}.csv"
        else:
            out_path = output_dir / f"{split}_{model_tag}.csv"
        write_rows_with_rationale(updated_rows, out_path)
        print(f"Wrote {len(updated_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

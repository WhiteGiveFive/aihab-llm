"""Choose the best habitat using Qwen3-VL with configurable prompt styles.

This script mirrors the batching/inference flow in `qwen3_vl_reasoning.py` and
supports two prompt modes:
1) `top3`: choose from the per-row top-3 L3 candidates with attributes.
2) `all_l3_names`: choose from the full L3 list (names only, no attributes).

It maps L3 ids using `REASSIGN_LABEL_NAME_L3` and pulls attributes from
`*_L3_ATTRS` via `get_habitat_attrs` when `top3` mode is used.

Outputs new CSVs (names differ by prompt mode):
  - data_tables/qwen3_vl_outputs/mis_Qwen_Qwen3-VL-4B-Instruct_choose_from_top3.csv
  - data_tables/qwen3_vl_outputs/correct_Qwen_Qwen3-VL-4B-Instruct_choose_from_top3.csv
  - data_tables/qwen3_vl_outputs/mis_Qwen_Qwen3-VL-4B-Instruct_choose_from_all_l3.csv
  - data_tables/qwen3_vl_outputs/correct_Qwen_Qwen3-VL-4B-Instruct_choose_from_all_l3.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from cs_hab import REASSIGN_LABEL_NAME_L3
from utils import (
    add_image_paths,          # reused for path attachment
    get_habitat_attrs,        # reused for L3 attribute lookup
    load_samples,             # reused for CSV reading
    safe_model_id,            # reused for output naming
    attach_rationales,        # reused for output saving
    write_rows_with_rationale, # reused for output saving
    load_qwen_and_processor, 
)

# ---- Constants (aligned with qwen3_vl_reasoning.py) ----
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_IMAGE_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/data/CS_Xplots_2019_2023_test"
)
DATA_TABLES_DIR = Path(
    "/home/hshi/Documents/researchproject/aihab/repo/aihab-llm/data_tables"
)
OUTPUT_DIR = DATA_TABLES_DIR / "qwen3_vl_outputs"


# ---- Data loading helpers ----
def load_split_rows(csv_path: Path, split: str, base_dir: Path):
    """Load a CSV split and attach image paths (reuse utils.add_image_paths)."""
    rows = load_samples(csv_path, split=split)
    return add_image_paths(rows, base_dir=base_dir)


def map_l3_id_to_name(l3_id: str) -> str:
    """Map an L3 numeric id (string in CSV) to name using REASSIGN_LABEL_NAME_L3."""
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
    """Build the forced-choice prompt for three candidate habitats.

    candidates: list of dicts, each with at least:
      - name: L3 habitat name
      - attrs: string (or short text derived from *_L3_ATTRS)
    """
    if len(candidates) != 3:
        raise ValueError(f"Expected 3 candidates, got {len(candidates)}.")
    header = (
        "You are given a ground-level habitat photo and three candidate habitat classes with typical visual descriptions. "
        "Analyse the photo with your ecology expertise and the given descriptions, and select exactly one habitat from the candidates that best match the photo."
    )
    payload = {"candidates": candidates}
    candidate_block = (
        f"Candidates and their descriptions(JSON): "
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
    """Build a forced-choice prompt using all L3 habitat names (no attributes)."""
    header = (
        "You are given a ground-level habitat photo and a list of candidate habitat classes. "
        "Select exactly one habitat that best matches the photo."
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


def build_messages(rows: List[Dict[str, str]], use_all_l3_names: bool = False):
    """Create chat messages with image + candidate prompt per row.

    Reuse the same message structure as qwen3_vl_reasoning.py.
    """
    def format_attrs(attrs: Optional[Dict[str, str]]) -> str:
        if not attrs:
            return "(not available)"
        return "; ".join(f"{key}: {value}" for key, value in attrs.items())

    all_l3_names = None
    if use_all_l3_names:
        all_l3_names = [name for _, name in sorted(REASSIGN_LABEL_NAME_L3.items())]

    all_messages = []
    for row in rows:
        image_path = Path(row["image_path"]).resolve()

        if use_all_l3_names:
            prompt = build_prompt_all_l3_names_only(all_l3_names)
        else:
            ids = [row["top3_label_1"], row["top3_label_2"], row["top3_label_3"]]
            names = [map_l3_id_to_name(l3_id) for l3_id in ids]
            attr_dicts = [get_habitat_attrs(name) for name in names]
            candidates = [
                {"name": name, "attrs": format_attrs(attrs)}
                for name, attrs in zip(names, attr_dicts)
            ]
            prompt = build_prompt_top3(candidates)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful ecologist."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        all_messages.append(messages)
    return all_messages


def batched_infer(
    model,
    processor,
    rows,
    batch_size=4,
    max_new_tokens=256,
    use_all_l3_names: bool = False,
):
    """Run batched inference (reuse logic from qwen3_vl_reasoning.py)."""
    # Ensure left padding for batch generation (Qwen3-VL guidance)
    processor.tokenizer.padding_side = "left"

    all_messages = build_messages(rows, use_all_l3_names=use_all_l3_names)

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
            max_new_tokens=max_new_tokens,
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
    """Print a few image paths for quick inspection (reuse logic from qwen3_vl_reasoning.py)."""
    if not image_paths:
        print(f"No image paths for {split}.")
        return
    count = min(sample_size, len(image_paths))
    print(f"Sample image paths for {split} (showing {count} of {len(image_paths)}):")
    for idx in range(count):
        print(f"  {idx + 1}. {image_paths[idx]}")


def print_sample_prompts(rows, split, sample_size=2, use_all_l3_names: bool = False):
    """Print a few constructed prompts for quick inspection."""
    if not rows:
        print(f"No rows for {split} to sample prompts.")
        return
    count = min(sample_size, len(rows))
    print(f"Sample prompts for {split} (showing {count} of {len(rows)}):")
    all_l3_names = None
    if use_all_l3_names:
        all_l3_names = [name for _, name in sorted(REASSIGN_LABEL_NAME_L3.items())]
    for idx in range(count):
        row = rows[idx]
        if use_all_l3_names:
            prompt = build_prompt_all_l3_names_only(all_l3_names)
        else:
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
        print(f"\n--- Prompt {idx + 1} ({row.get('file_name', 'unknown')}): ---")
        print(prompt)


# ---- Output helpers ----
# def attach_predictions(rows, pred_label_names, rationales):
#     """Attach predicted label names + rationales to rows.

#     If you want, extend utils.write_rows_with_rationale or add a new writer here.
#     """
#     if len(rows) != len(pred_label_names):
#         raise ValueError(
#             f"Row count {len(rows)} != pred_label_names count {len(pred_label_names)}"
#         )
#     if len(rows) != len(rationales):
#         raise ValueError(
#             f"Row count {len(rows)} != rationale count {len(rationales)}"
#         )
#     updated = []
#     for row, pred_label_name, rationale in zip(rows, pred_label_names, rationales):
#         new_row = dict(row)
#         new_row["LLM_pred_habitat"] = pred_label_name
#         new_row["rationale"] = rationale
#         updated.append(new_row)
#     return updated


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (mirror qwen3_vl_reasoning.py)."""
    parser = argparse.ArgumentParser(
        description="Choose best habitat from top-3 candidates using Qwen3-VL."
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
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per sample.",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 weights if supported by your hardware.",
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
        "--prompt-mode",
        type=str,
        default="top3",
        choices=("top3", "all_l3_names"),
        help="Prompt style: top3 candidates with attributes, or all L3 names only.",
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
    # Optional: add sample prompt/paths flags, mirroring qwen3_vl_reasoning.py
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, processor = load_qwen_and_processor(args.model_id, args.bfloat16)
    model_tag = safe_model_id(args.model_id)

    base_image_dir = args.base_image_dir
    data_tables_dir = args.data_tables_dir
    output_dir = data_tables_dir / "qwen3_vl_outputs"

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
                use_all_l3_names=(args.prompt_mode == "all_l3_names"),
            )

        outputs = batched_infer(
            model,
            processor,
            rows,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_all_l3_names=(args.prompt_mode == "all_l3_names"),
        )
        updated_rows = attach_rationales(rows, outputs)

        suffix = "choose_from_top3" if args.prompt_mode == "top3" else "choose_from_all_l3"
        out_path = output_dir / f"{split}_{model_tag}_{suffix}.csv"
        write_rows_with_rationale(updated_rows, out_path)
        print(f"Wrote {len(updated_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

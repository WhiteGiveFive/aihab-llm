"""Utility functions for the Qwen3-VL reasoning script."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Dict, Optional
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, Glm4vForConditionalGeneration, AutoProcessor
import torch

import cs_hab
from PIL import Image


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader)


def derive_image_path(base_dir: Path, file_name: str) -> Path:
    """Derive a full image path from a file name."""
    return base_dir / file_name


def add_image_paths(
    rows: Iterable[Dict[str, str]],
    base_dir: Path,
    file_name_column: str = "file_name",
    output_column: str = "image_path",
    resolve: bool = True,
) -> List[Dict[str, str]]:
    """Attach derived image paths to each row under output_column."""
    base_dir = base_dir.resolve() if resolve else base_dir
    updated: List[Dict[str, str]] = []
    for row in rows:
        if file_name_column not in row:
            raise KeyError(f"Missing column '{file_name_column}' in row: {row}")
        file_name = row[file_name_column]
        image_path = derive_image_path(base_dir, file_name)
        updated_row = dict(row)
        updated_row[output_column] = str(image_path)
        updated.append(updated_row)
    return updated


def load_samples(csv_path: Path, split: Optional[str] = None) -> List[Dict[str, str]]:
    """Load a CSV and optionally tag each row with a split label."""
    rows = read_csv_rows(csv_path)
    if split is None:
        return rows
    tagged: List[Dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        updated["split"] = split
        tagged.append(updated)
    return tagged


def load_mis_and_correct(
    mis_csv: Path, correct_csv: Path, base_dir: Path
) -> List[Dict[str, str]]:
    """Load mis/correct tables, tag splits, and attach image paths."""
    mis_rows = load_samples(mis_csv, split="mis")
    correct_rows = load_samples(correct_csv, split="correct")
    combined = mis_rows + correct_rows
    return add_image_paths(combined, base_dir)


def write_rows_with_rationale(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if "rationale" not in fieldnames:
        fieldnames.append("rationale")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attach_rationales(rows, rationales):
    if len(rows) != len(rationales):
        raise ValueError(f"Row count {len(rows)} != rationale count {len(rationales)}")
    updated = []
    for row, rationale in zip(rows, rationales):
        new_row = dict(row)
        new_row["rationale"] = rationale
        updated.append(new_row)
    return updated


def safe_model_id(model_id):
    return model_id.replace("/", "_").replace(":", "_")


def get_habitat_attrs(label: str) -> Optional[Dict[str, str]]:
    """Return the attribute dict for a habitat label, or None if not found."""
    attr_maps = [
        cs_hab.GRASSLAND_L3_ATTRS,
        cs_hab.WETLAND_L3_ATTRS,
        cs_hab.HEATHLAND_L3_ATTRS,
        cs_hab.CROPLAND_L3_ATTRS,
        cs_hab.WOODLAND_L3_ATTRS,
        cs_hab.MARINE_L3_ATTRS,
        cs_hab.MONTANE_L3_ATTRS,
        cs_hab.RIVERS_L3_ATTRS,
        cs_hab.SPARSE_L3_ATTRS,
        cs_hab.URBAN_L3_ATTRS,
        cs_hab.SEA_L3_ATTRS,
    ]
    for attr_map in attr_maps:
        if label in attr_map:
            return attr_map[label]
    return None


def load_image(image_path: Path) -> Image.Image:
    """Load an image as RGB for model input."""
    with Image.open(image_path) as img:
        return img.convert("RGB")


def prepare_inputs(processor, image_path: Path, prompt: str):
    """Prepare model inputs; adjust for chat templates if available. Used for single image inference."""
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


# Modules for model loading
MODEL_ZOO = {
    'qwen': {
        'dense': ['Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-4B-Thinking', 'Qwen/Qwen3-VL-8B-Instruct', 
        'Qwen/Qwen3-VL-8B-Thinking', 'Qwen/Qwen3-VL-32B-Instruct', 'Qwen/Qwen3-VL-32B-Thinking'], 
        'moe': ['Qwen/Qwen3-VL-30B-A3B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Thinking']
        }, 
    'mistral': {
        'ministral': [
            'mistralai/Ministral-3-3B-Instruct-2512', 
            'mistralai/Ministral-3-8B-Instruct-2512', 
            'mistralai/Ministral-3-14B-Instruct-2512', 
            'mistralai/Ministral-3-3B-Reasoning-2512', 
            'mistralai/Ministral-3-8B-Reasoning-2512', 
            'mistralai/Ministral-3-14B-Reasoning-2512'
            ]
    }, 
    'zai-org': {
        'GLM-4.6V': ['zai-org/GLM-4.6V-Flash']
    }
}


def select_qwen3_generator(model_id: str):
    model_id = model_id.strip()
    if model_id in set(MODEL_ZOO.get("qwen", {}).get("moe", [])):
        return Qwen3VLMoeForConditionalGeneration
    if model_id in set(MODEL_ZOO.get("qwen", {}).get("dense", [])):
        return Qwen3VLForConditionalGeneration
    if model_id in set(MODEL_ZOO.get("zai-org", {}).get("GLM-4.6V", [])):
        return Glm4vForConditionalGeneration
    raise ValueError(
        f"Unknown Qwen or GLM model_id: {model_id}. "
        f"Please add it to utils.MODEL_ZOO."
    )

def load_qwen_and_processor(model_id: str, use_bfloat16: bool):
    """Load the HF processor and model. Adjust classes here if needed."""
    # NOTE: Qwen3-VL may require a specific model class from transformers.
    # If AutoModelForCausalLM fails, replace it with the recommended class from HF.

    dtype = torch.bfloat16 if use_bfloat16 else "auto"
    processor = AutoProcessor.from_pretrained(model_id)
    model_generator = select_qwen3_generator(model_id)
    model = model_generator.from_pretrained(
        model_id, dtype=dtype, device_map="auto"
        )

    return model, processor
    
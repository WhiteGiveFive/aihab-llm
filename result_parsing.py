"""
Parse LLM responses and compute accuracy for Qwen3 top-3 selection outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

def parse_model_outputs(
    outputs: List[str],
    candidate_names: List[List[str]],
) -> tuple[List[str], List[str]]:
    """Parse strict JSON outputs into predicted habitat names and rationales.

    Expected format per item:
      {"pred_habitat":"<one of candidates>","rationale":"<short text>"}
    """
    if len(outputs) != len(candidate_names):
        raise ValueError(
            f"Output count {len(outputs)} != candidate list count {len(candidate_names)}"
        )
    pred_habitats: List[str] = []
    rationales: List[str] = []
    for idx, text in enumerate(outputs):
        raw = text.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON at index {idx}: {raw!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Output at index {idx} is not a JSON object: {raw!r}")
        if "pred_habitat" not in parsed or "rationale" not in parsed:
            raise ValueError(
                f"Missing required keys at index {idx}: {raw!r}"
            )
        pred = parsed["pred_habitat"]
        rationale = parsed["rationale"]
        if not isinstance(pred, str) or not isinstance(rationale, str):
            raise ValueError(
                f"Values must be strings at index {idx}: {raw!r}"
            )
        expected = candidate_names[idx]
        if pred not in expected:
            raise ValueError(
                f"Pred habitat {pred!r} not in candidates at index {idx}: {expected}"
            )
        pred_habitats.append(pred)
        rationales.append(rationale)
    return pred_habitats, rationales


def read_csv_rows(csv_path: Path) -> List[dict]:
    """Read a CSV file into a list of row dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader)


def extract_pred_candidate(raw_text: str, row_idx: int, csv_path: Path) -> str:
    """Extract pred_candidate from a JSON string in the rationale column."""
    raw = (raw_text or "").strip()
    if raw == "":
        raise ValueError(f"Empty rationale at row {row_idx} in {csv_path}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON at row {row_idx} in {csv_path}: {raw!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Rationale is not a JSON object at row {row_idx} in {csv_path}: {raw!r}"
        )
    if "pred_candidate" in parsed:
        pred = parsed["pred_candidate"]
    elif "pred_habitat" in parsed:
        pred = parsed["pred_habitat"]
    else:
        raise ValueError(
            f"Missing pred_candidate at row {row_idx} in {csv_path}: {raw!r}"
        )
    if not isinstance(pred, str):
        raise ValueError(
            f"pred_candidate is not a string at row {row_idx} in {csv_path}: {raw!r}"
        )
    return pred.strip()


def score_rows(rows: List[dict], csv_path: Path) -> tuple[int, int]:
    """Return (total, correct) for a list of rows."""
    total = 0
    correct = 0
    for idx, row in enumerate(rows):
        gt = (row.get("ground_truth_word_label") or "").strip()
        if gt == "":
            raise ValueError(f"Missing ground_truth_word_label at row {idx} in {csv_path}")
        pred = extract_pred_candidate(row.get("rationale", ""), idx, csv_path)
        total += 1
        if pred == gt:
            correct += 1
    return total, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy from Qwen3 top-3 selection CSVs."
    )
    parser.add_argument(
        "--mis",
        type=Path,
        default=Path(
            "data_tables/qwen3_vl_outputs/"
            "mis_Qwen_Qwen3-VL-4B-Instruct_choose_from_top3.csv"
        ),
        help="Path to the mis split CSV with rationale JSON.",
    )
    parser.add_argument(
        "--correct",
        type=Path,
        default=Path(
            "data_tables/qwen3_vl_outputs/"
            "correct_Qwen_Qwen3-VL-4B-Instruct_choose_from_top3.csv"
        ),
        help="Path to the correct split CSV with rationale JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mis_rows = read_csv_rows(args.mis)
    correct_rows = read_csv_rows(args.correct)

    mis_total, mis_correct = score_rows(mis_rows, args.mis)
    correct_total, correct_correct = score_rows(correct_rows, args.correct)

    total = mis_total + correct_total
    correct = mis_correct + correct_correct
    accuracy = (correct / total) if total else 0.0

    print(f"mis: {mis_correct}/{mis_total} = {mis_correct / mis_total:.4f}")
    print(f"correct: {correct_correct}/{correct_total} = {correct_correct / correct_total:.4f}")
    print(f"combined: {correct}/{total} = {accuracy:.4f}")


if __name__ == "__main__":
    main()

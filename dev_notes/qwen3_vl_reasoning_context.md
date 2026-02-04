# Qwen3-VL reasoning script context

Date: 2026-02-02

## Goal
Run Qwen3-VL on local CS photos in
`/home/hshi/Documents/researchproject/aihab/repo/aihab-clip/data/CS_Xplots_2019_2023_test`
with a Python batch script (`qwen3_vl_reasoning.py`) that:
- uses per-row prompts referencing ground-truth labels and habitat attributes
- runs batch inference for mis/correct CSVs
- writes outputs to `data_tables/qwen3_vl_outputs/`

## Current script status (high level)
- Uses `Qwen3VLForConditionalGeneration` + `AutoProcessor`.
- Batch-only flow (single-image path removed from main).
- Builds per-row chat-style `messages` with a system role.
- Per-row prompt includes ground-truth class + habitat attributes and requests JSON output.
- Generates output with `model.generate` and trims prompt tokens.
- Outputs written to `data_tables/qwen3_vl_outputs/{split}_{MODEL_ID}.csv`.

## Key decisions and fixes made
- **Model class**: switched to `Qwen3VLForConditionalGeneration` (official instruction) instead of `AutoModelForCausalLM`.
- **Image path handling**:
  - `file://` URIs caused `Incorrect image source` errors.
  - Passing a **plain absolute file path string** in `messages` works.
  - Code uses `str(image_path.resolve())` for `messages`.
- **Batch inference**:
  - Built `batched_infer` using `apply_chat_template` with padding and left padding for tokenizer.
  - Runs mis and correct separately.
  - Added CLI flags for `--batch-size`, `--sample-paths`, and `--sample-prompts`.
- **Prompt design**:
  - Per-row prompt now includes ground-truth label and habitat attributes from `cs_hab.py`.
  - JSON-only response format enforced: `{"score": <1-5>, "rationale": "<short text>"}`.
  - System role added: `"You are a helpful ecologist."`
- **Outputs**:
  - New helper functions in `utils.py` for CSV IO and naming: `attach_rationales`, `write_rows_with_rationale`, `safe_model_id`.
  - Output folder: `data_tables/qwen3_vl_outputs/`.

## Known nuances
- There was discussion about `dtype` vs `torch_dtype` in `from_pretrained`.
  - The Qwen model card shows `dtype=...` usage.
  - The script currently uses `dtype` per the model card; this is acceptable for the installed Transformers build.
- `write_rows_with_rationale` assumes non-empty rows (uses `rows[0]`).

## Last change
- Switched to batch-only pipeline and removed single-image path from main.
- Added prompt sampling and path sampling printouts.
- Added per-row prompt with habitat attributes and JSON-only format.

## How to run
Example:
```bash
python qwen3_vl_reasoning.py --batch-size 2 --sample-paths 2 --sample-prompts 1 --bfloat16
```

## Open items
- If needed, verify the `apply_chat_template` + local file path flow against future library updates.
- Consider making `write_rows_with_rationale` robust to empty CSVs.

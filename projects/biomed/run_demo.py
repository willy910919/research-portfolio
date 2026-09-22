"""Synthetic-only demo of actual import and data comparison components."""
import json
from pathlib import Path
from research_room.dataset_import import read_dataset
from research_room.data_inspection import compare_frames

root = Path(__file__).resolve().parent
before, metadata = read_dataset(root / "examples" / "synthetic_four_rows.csv", missing_policy="blank_only")
after = before.loc[before["INTERPRETATION"].notna()].copy()
comparison = compare_frames(before, after)
checks = {
    "four_to_two_rows": len(before) == 4 and len(after) == 2,
    "leading_zero_ids_preserved": list(after["ID"]) == ["0003", "0004"],
    "literal_NA_and_nan_preserved": list(before["CATEGORY"].iloc[:2]) == ["NA", "nan"],
    "filtered_mean_35": float(after["BMI"].mean()) == 35.0,
    "retained_values_unchanged": all(f["changed_cells"] == 0 for f in comparison["fields"]),
}
result = {
    "synthetic_data_only": True,
    "source_version": "0.14.4",
    "input_rows": len(before), "output_rows": len(after),
    "retained_ids": list(after["ID"]),
    "mean_bmi": float(after["BMI"].mean()),
    "checks": checks,
    "limitations": "This demonstrates import and data provenance, not the full UI, LLM routing, or clinical validity.",
}
assert all(checks.values()), result
print(json.dumps(result, ensure_ascii=False, indent=2))

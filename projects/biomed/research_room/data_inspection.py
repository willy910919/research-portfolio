"""Owner-only, immutable receipts for tool inputs and dataset changes."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import re
from urllib.parse import quote
import uuid

import pandas as pd


def scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def fingerprint(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(c), str(d)) for c, d in frame.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def compare_frames(before, after, limit=100):
    """Align by preserved row identity, never by post-filter row position."""
    aligned = before.index.is_unique and after.index.is_unique
    common = before.index.intersection(after.index, sort=False) if aligned else []
    removed = before.index.difference(after.index, sort=False) if aligned else []
    added = after.index.difference(before.index, sort=False) if aligned else []
    fields, samples = [], []
    for column in dict.fromkeys([*before.columns, *after.columns]):
        left = before[column] if column in before else None
        right = after[column] if column in after else None
        changed = None
        if before is after and aligned:
            changed = 0
        elif aligned and left is not None and right is not None:
            a, b = left.loc[common], right.loc[common]
            equal = (a.eq(b).fillna(False) | (a.isna() & b.isna()))
            changed = int((~equal).sum())
            for row in common[~equal.to_numpy(dtype=bool)][:3]:
                if len(samples) < limit:
                    samples.append({"kind": "modified", "field": str(column),
                                    "input_row": int(before.index.get_loc(row)) + 1,
                                    "output_row": int(after.index.get_loc(row)) + 1,
                                    "before": scalar(left.loc[row]), "after": scalar(right.loc[row])})
        fields.append({"field": str(column),
                       "dtype_before": str(left.dtype) if left is not None else None,
                       "dtype_after": str(right.dtype) if right is not None else None,
                       "missing_before": int(left.isna().sum()) if left is not None else None,
                       "missing_after": int(right.isna().sum()) if right is not None else None,
                       "changed_cells": changed,
                       "status": "added" if left is None else "removed" if right is None else
                       "modified" if changed or str(left.dtype) != str(right.dtype) else "unchanged"})
    # Show a bounded subset of columns for removed/added/retained row examples.
    preview_columns = list(dict.fromkeys([f["field"] for f in fields if f["status"] != "unchanged"]
                                        + list(before.columns) + list(after.columns)))[:8]
    for kind, rows in (("removed", removed[:3]), ("added", added[:3]), ("retained", common[:3])):
        for row in rows:
            for column in preview_columns:
                if len(samples) >= limit:
                    break
                has_before = row in before.index and column in before
                has_after = row in after.index and column in after
                samples.append({"kind": kind, "field": str(column),
                                "input_row": int(before.index.get_loc(row)) + 1 if has_before else None,
                                "output_row": int(after.index.get_loc(row)) + 1 if has_after else None,
                                "before": scalar(before.at[row, column]) if has_before else None,
                                "after": scalar(after.at[row, column]) if has_after else None})
    return {"row_alignment": "preserved_index" if aligned else "unavailable",
            "rows_removed": len(removed) if aligned else None,
            "rows_added": len(added) if aligned else None,
            "fields": fields, "samples": samples, "sample_limit": limit,
            "preview_column_limit": 8}


class DataInspectionStore:
    def __init__(self, artifact_root, session_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("Invalid session id")
        self.directory = Path(artifact_root) / session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = "/api/artifacts/" + quote(session_id) + "/"
        self.snapshots = {}

    def snapshot(self, frame):
        key = fingerprint(frame)
        if key in self.snapshots:
            return self.snapshots[key]
        name = "data_snapshot_" + uuid.uuid4().hex + ".csv"
        path = self.directory / name
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            frame.to_csv(stream, index=False)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        schema = {"columns": [{"name": str(c), "dtype": str(d)} for c, d in frame.dtypes.items()],
                  "missing_policy": "blank_cells", "encoding": "utf-8-sig",
                  "note": "CSV does not preserve types; use this schema. Literal NA/nan are not null markers."}
        schema_name = name.replace(".csv", ".schema.json")
        self._json(schema_name, schema)
        result = {"rows": len(frame), "columns": len(frame.columns),
                  "missing_cells": int(frame.isna().sum().sum()),
                  "fingerprint": key, "sha256": digest.hexdigest(),
                  "download_url": self.prefix + name, "schema_url": self.prefix + schema_name}
        self.snapshots[key] = result
        return result

    def _json(self, name, value):
        temporary = self.directory / (uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, self.directory / name)

    def record(self, before, after, *, tool, run_id="", node_id="", scope="analysis_input",
               source_node_id="", input_snapshot=None, status="completed", result_download_url=""):
        receipt_id = "DI_" + uuid.uuid4().hex
        input_snapshot = input_snapshot or (self.snapshot(before) if before is not None else None)
        if before is not None and input_snapshot and fingerprint(before) != input_snapshot["fingerprint"]:
            raise ValueError("Input mutated during execution; comparison cannot be verified")
        output_snapshot = (input_snapshot if after is before else self.snapshot(after)) if after is not None and status == "completed" else None
        comparison = compare_frames(before, after) if before is not None and output_snapshot else {"fields": [], "samples": []}
        changed = bool(input_snapshot and output_snapshot and input_snapshot["fingerprint"] != output_snapshot["fingerprint"])
        receipt = {"schema_version": "1.0", "inspection_id": receipt_id,
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "tool": tool, "run_id": run_id, "node_id": node_id,
                   "source_node_id": source_node_id, "scope": scope, "status": status,
                   "change_kind": "failed" if status != "completed" else "not_applicable" if before is None
                   else "changed" if changed else "unchanged",
                   "before": input_snapshot, "after": output_snapshot, **comparison}
        if result_download_url.startswith(self.prefix):
            receipt["result_download_url"] = result_download_url
        field_name, sample_name = receipt_id + "_fields.csv", receipt_id + "_samples.csv"
        pd.DataFrame(receipt["fields"]).to_csv(self.directory / field_name, index=False, encoding="utf-8-sig")
        pd.DataFrame(receipt["samples"]).to_csv(self.directory / sample_name, index=False, encoding="utf-8-sig")
        receipt["downloads"] = {"fields": self.prefix + field_name, "samples": self.prefix + sample_name,
                                "report": self.prefix + receipt_id + ".json"}
        self._json(receipt_id + ".json", receipt)
        return self.public(receipt)

    @staticmethod
    def public(receipt):
        # No values or private preview content may enter agent events/memory.
        return {key: receipt.get(key) for key in ("inspection_id", "created_at", "tool", "run_id", "node_id",
                                                 "source_node_id", "scope", "status", "change_kind")}

    def get(self, inspection_id):
        if not re.fullmatch(r"DI_[a-f0-9]{32}", inspection_id):
            raise FileNotFoundError("Unknown inspection")
        return json.loads((self.directory / (inspection_id + ".json")).read_text(encoding="utf-8"))

    def list(self):
        return sorted((self.public(json.loads(p.read_text(encoding="utf-8")))
                       for p in self.directory.glob("DI_*.json")), key=lambda r: r["created_at"], reverse=True)

#!/usr/bin/env python3
"""
Build a Piper voice annotation sidecar from a LibriTTS-P style XLSX workbook.

Writes ``<model>.onnx.speakers.json`` next to the ONNX so middleware can merge
annotations into /voices speaker rows.

Annotation priority (first non-blank wins): Annotator 3 → Annotator 1 → Annotator 2.

Example:
  python scripts/build_speaker_annotation_sidecar.py ^
    --xlsx clean100_voice_descriptors.xlsx ^
    --onnx-json "Z:/Voices/en/en_US/Licensed/Clean 100/medium/clean100.onnx.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ANNOTATOR_PRIORITY = ("Annotator 3", "Annotator 1", "Annotator 2")


def _cell(row: tuple, idx: dict[str, int], key: str) -> str:
    if key not in idx:
        return ""
    value = row[idx[key]]
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return text


def pick_annotation(row: tuple, idx: dict[str, int]) -> str:
    for key in ANNOTATOR_PRIORITY:
        text = _cell(row, idx, key)
        if text:
            return text
    return ""


def load_speaker_id_map(onnx_json: Path) -> dict[int, str]:
    data = json.loads(onnx_json.read_text(encoding="utf-8"))
    raw = data.get("speaker_id_map") or data.get("speakerIdMap") or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"No speaker_id_map in {onnx_json}")
    out: dict[int, str] = {}
    for name, sid in raw.items():
        try:
            out[int(sid)] = str(name).strip()
        except (TypeError, ValueError):
            continue
    if not out:
        raise SystemExit(f"Empty speaker_id_map in {onnx_json}")
    return out


def build_sidecar(xlsx: Path, onnx_json: Path) -> dict:
    try:
        import openpyxl
    except ImportError as ex:
        raise SystemExit("openpyxl is required: pip install openpyxl") from ex

    by_id = load_speaker_id_map(onnx_json)
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    if not rows:
        raise SystemExit("XLSX has no rows")

    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    idx = {name: i for i, name in enumerate(header) if name}
    required = ("Piper Speaker ID",)
    for key in required:
        if key not in idx:
            raise SystemExit(f"XLSX missing required column: {key}")

    speakers: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    skipped_unknown: list[int] = []
    for row in rows[1:]:
        sid_raw = row[idx["Piper Speaker ID"]]
        if sid_raw is None or str(sid_raw).strip() == "":
            continue
        sid = int(sid_raw)
        if sid not in by_id:
            # XLSX rows for ids the model cannot synthesize must not enter the sidecar.
            skipped_unknown.append(sid)
            continue
        friendly = _cell(row, idx, "Friendly Name")
        annotation = pick_annotation(row, idx)
        json_name = by_id[sid]
        if friendly and json_name != friendly:
            mismatches.append(f"id={sid} xlsx={friendly!r} json={json_name!r}")
        entry: dict[str, str] = {
            "name": json_name or friendly or f"speaker_{sid}",
        }
        if annotation:
            entry["annotation"] = annotation
        speakers[str(sid)] = entry

    if skipped_unknown:
        print(
            f"WARNING: skipped {len(skipped_unknown)} XLSX id(s) not in speaker_id_map "
            f"(examples: {skipped_unknown[:10]})",
            file=sys.stderr,
        )

    if mismatches:
        print(f"WARNING: {len(mismatches)} name mismatches vs onnx.json", file=sys.stderr)
        for line in mismatches[:10]:
            print(f"  {line}", file=sys.stderr)

    model_name = onnx_json.name
    if model_name.lower().endswith(".json"):
        # clean100.onnx.json → clean100.onnx
        model_name = model_name[: -len(".json")]

    return {
        "version": 1,
        "model": model_name,
        "source": xlsx.name,
        "annotation_priority": list(ANNOTATOR_PRIORITY),
        "speakers": speakers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", required=True, type=Path, help="Voice descriptor workbook")
    parser.add_argument(
        "--onnx-json",
        required=True,
        type=Path,
        help="Piper companion JSON (e.g. clean100.onnx.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <onnx>.speakers.json beside the model)",
    )
    args = parser.parse_args()

    onnx_json = args.onnx_json
    if not onnx_json.is_file():
        raise SystemExit(f"Missing onnx json: {onnx_json}")
    if not args.xlsx.is_file():
        raise SystemExit(f"Missing xlsx: {args.xlsx}")

    # clean100.onnx.json → clean100.onnx.speakers.json
    stem_onnx = Path(str(onnx_json)[: -len(".json")] if str(onnx_json).lower().endswith(".json") else onnx_json)
    out = args.out or Path(str(stem_onnx) + ".speakers.json")

    payload = build_sidecar(args.xlsx, onnx_json)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(payload['speakers'])} speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

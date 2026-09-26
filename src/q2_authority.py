"""Create a Q2 authority NPZ by exact indexing of a numerically accepted candidate.

No PDE solution, spatial/temporal interpolation, rounding, clipping or new FV
state is constructed. The numerical reviewer must accept this exact candidate
first. This program preserves the original candidate description and both source
trajectories' identities, including the limitations of Richardson point outputs.
It does not produce an Excel file or issue the final export acceptance gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
BASE = ROOT / "records/q2-full-20260911"
POSTPROCESSOR = ROOT / "src/q2_grid_extrapolation.py"
KEYS = ("time_s", "radius_m", "T_C", "C_kg_kg")
EXPECTED_TIMES = np.arange(10801, dtype=np.float64)
EXPECTED_RADII = np.arange(21, dtype=np.float64) * .001


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path leaves this project: {path}")
    return path


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def check_python() -> None:
    config = read_json(ROOT / "project_config.json")
    expected = Path(os.path.expandvars(config["python"])).resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError(f"Use the configured project Python: {expected}")


def verify_hash(path: Path, expected, label: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Invalid {label} SHA-256")
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    actual = sha256(path)
    if actual != expected.lower():
        raise ValueError(f"Changed {label}: {path}")
    return actual


def check_analysis_verdict(analysis: dict, candidate_sha: str) -> None:
    """Only a verdict on this exact candidate is allowed; booleans are strict."""
    if analysis.get("numerical_pass") is not True:
        raise ValueError("Analysis must explicitly state numerical_pass=true")
    count = analysis.get("unresolved_rounding_count")
    if type(count) is not int or count != 0:
        raise ValueError("Analysis must state integer unresolved_rounding_count=0")
    if analysis.get("accepted_candidate_sha256") != candidate_sha:
        raise ValueError("Analysis has not accepted this candidate's exact NPZ SHA-256")


def select_integer_rows(times: np.ndarray) -> np.ndarray:
    """Use exact equality only; near-integer samples never replace integer nodes."""
    if times.dtype != np.float64 or times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("Candidate times must be a finite binary64 vector")
    if len(times) < len(EXPECTED_TIMES) or times[0] != 0 or times[-1] != 10800:
        raise ValueError("Candidate must cover exactly 0..10800 s, with all integer nodes")
    if not np.all(np.diff(times) > 0):
        raise ValueError("Candidate times contain duplicates or are not strictly ordered")
    indices = np.flatnonzero(times == np.floor(times)).astype(np.int64, copy=False)
    if not np.array_equal(times[indices], EXPECTED_TIMES):
        raise ValueError("Missing or wrong integer nodes; interpolation and invented samples are forbidden")
    return indices


def extract_arrays(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    with np.load(path, allow_pickle=False) as source:
        if not set(KEYS).issubset(source.files):
            raise ValueError("Candidate NPZ lacks one of time_s/radius_m/T_C/C_kg_kg")
        if "fixture_only" in source.files:
            raise ValueError("A fixture NPZ can never become a real authority")
        data = {key: source[key].copy() for key in KEYS}
    if any(array.dtype != np.float64 for array in data.values()):
        raise ValueError("Candidate arrays must already be binary64; no silent dtype conversion")
    times, radii = data["time_s"], data["radius_m"]
    indices = select_integer_rows(times)
    if radii.shape != (21,) or not np.array_equal(radii, EXPECTED_RADII):
        raise ValueError("Candidate must already contain exact 0..0.020 m radii by 0.001 m")
    for key, initial in (("T_C", 28.), ("C_kg_kg", 2.55)):
        array = data[key]
        if array.shape != (len(times), 21) or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid dimensions or non-finite values: {key}")
        if not np.all(array[0] == initial):
            raise ValueError(f"Uniform original t=0 state was altered: {key}")
    if np.any(data["C_kg_kg"] <= 0) or np.any(data["T_C"] + 273.15 <= 0):
        raise ValueError("Candidate violates positive moisture or absolute-temperature domain")
    if np.any(data["T_C"] < 0):
        raise ValueError("Candidate is incompatible with the current Q2 nonnegative Celsius export domain")
    output = {"time_s": times[indices].copy(), "radius_m": radii.copy(),
              "T_C": data["T_C"][indices, :].copy(),
              "C_kg_kg": data["C_kg_kg"][indices, :].copy()}
    for key in KEYS:
        expected = data[key] if key == "radius_m" else data[key][indices]
        if not np.array_equal(output[key], expected) or output[key].tobytes() != expected.tobytes():
            raise AssertionError(f"Index/copy unexpectedly changed source values: {key}")
    return output, indices, len(times)


def check_sources(candidate_folder: Path, analysis_path: Path) -> tuple[dict, dict, dict]:
    """Bind the candidate description, source files, postprocessor and analysis."""
    solution = candidate_folder / "solution.npz"
    description_path = candidate_folder / "candidate.json"
    description = read_json(description_path)
    if description.get("question") != "Q2":
        raise ValueError("Candidate question must be Q2")
    candidate_sha = verify_hash(solution, description.get("solution_sha256"), "candidate solution")
    verify_hash(POSTPROCESSOR, description.get("postprocessor_sha256"), "candidate postprocessor")
    sources = description.get("sources")
    if not isinstance(sources, dict) or len(sources) != 2:
        raise ValueError("The uniform Richardson candidate must identify exactly two source NPZ files")
    cells = description.get("source_grid_cells")
    if (not isinstance(cells, list) or len(cells) != 2 or any(type(n) is not int or n <= 0 for n in cells)
            or cells[1] != 2 * cells[0]):
        raise ValueError("Candidate must identify two source grid sizes with a ratio of 2")
    for key in ("no_per_point_branch", "no_clipping", "no_intermediate_decimal_rounding", "physical_model_unchanged"):
        if description.get(key) is not True:
            raise ValueError(f"Candidate description lacks its confirmed condition: {key}")
    if description.get("extrapolation_order_assumption") != 2 or description.get("fourth_order_claim") is not False:
        raise ValueError("Unexpected Richardson order description")
    bindings = {str(solution): candidate_sha, str(description_path): sha256(description_path),
                str(POSTPROCESSOR): sha256(POSTPROCESSOR), str(SCRIPT): sha256(SCRIPT),
                str(ROOT / "project_config.json"): sha256(ROOT / "project_config.json")}
    normalized = set()
    for value, expected in sources.items():
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Source NPZ paths must be absolute project paths")
        source = project_path(value)
        if source.suffix.lower() != ".npz" or source in normalized or source == solution:
            raise ValueError("Source NPZ identities must be distinct from each other and from the candidate")
        normalized.add(source)
        bindings[str(source)] = verify_hash(source, expected, "source trajectory NPZ")
        # Include existing run records as provenance without claiming a new FV state.
        run_record = source.parent / "run.json"
        if run_record.is_file():
            run = read_json(run_record)
            if run.get("solution_sha256") != expected.lower():
                raise ValueError(f"Source run.json does not match its NPZ: {source}")
            bindings[str(run_record)] = sha256(run_record)
    analysis = read_json(analysis_path)
    check_analysis_verdict(analysis, candidate_sha)
    bindings[str(analysis_path)] = sha256(analysis_path)
    return description, analysis, bindings


def verify_output(path: Path, expected: dict[str, np.ndarray], metadata_text: str) -> dict:
    with np.load(path, allow_pickle=False) as result:
        if set(result.files) != {*KEYS, "metadata_json"}:
            raise AssertionError("Authority keys changed during serialization")
        for key in KEYS:
            actual = result[key]
            if (actual.dtype != expected[key].dtype or not np.array_equal(actual, expected[key])
                    or actual.tobytes() != expected[key].tobytes()):
                raise AssertionError(f"Serialized authority differs from exact candidate copy: {key}")
        text = result["metadata_json"]
        if text.shape != () or text.dtype.kind != "U" or text.item() != metadata_text:
            raise AssertionError("Authority metadata was not preserved exactly")
    return {"bitwise_copied_arrays": list(KEYS), "metadata_text_preserved": True,
            "time_count": 10801, "radius_count": 21, "physical_values_copied": 453642,
            "submitted_physical_values": 453600, "initial_state_included": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Folder containing solution.npz and candidate.json")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--preflight", action="store_true", help="Read and verify only; create no files")
    args = parser.parse_args()
    check_python()
    candidate, analysis_path, output, record = map(project_path, (args.candidate, args.analysis, args.output, args.record))
    if not candidate.is_relative_to(BASE / "candidates") or not candidate.is_dir():
        raise ValueError("Candidate must be an existing Q2 candidate directory")
    if not output.is_relative_to(ROOT / "results/q2") or output.suffix.lower() != ".npz":
        raise ValueError("Authority output must be an NPZ under results/q2")
    if not record.is_relative_to(BASE) or record.suffix.lower() != ".json":
        raise ValueError("Authority creation record must be a JSON under Q2 full-run records")
    if output.exists() or record.exists():
        raise FileExistsError("Authority output or creation record already exists; refusing overwrite")
    description, analysis, source_hashes = check_sources(candidate, analysis_path)
    arrays, indices, candidate_time_count = extract_arrays(candidate / "solution.npz")
    selection = {"rule": "Exact integer-coordinate equality followed by array indexing/copy; no interpolation",
                 "candidate_time_count": candidate_time_count,
                 "source_time_row_indices": indices.tolist(), "source_radius_column_indices": list(range(21)),
                 "output_time_s": {"start": 0, "stop": 10800, "step": 1},
                 "output_radius_m": {"start": 0, "stop": .02, "step": .001}}
    metadata = {"question": "Q2", "authority_kind": "uniform_spatial_Richardson_point_outputs",
        "authority_status": "INDEXED_FROM_NUMERICALLY_ACCEPTED_CANDIDATE",
        "candidate_description_verbatim": description,
        "candidate_description_path": str(candidate / "candidate.json"),
        "candidate_solution_path": str(candidate / "solution.npz"),
        "candidate_solution_sha256": source_hashes[str(candidate / "solution.npz")],
        "source_trajectory_sha256": description["sources"], "source_grid_cells": description["source_grid_cells"],
        "units": {"time_s": "s", "radius_m": "m", "T_C": "degree C", "C_kg_kg": "dry-basis kg/kg"},
        "analysis_path": str(analysis_path), "analysis_sha256": source_hashes[str(analysis_path)],
        "analysis_verdict": {k: analysis[k] for k in ("numerical_pass", "unresolved_rounding_count", "accepted_candidate_sha256")},
        "generator_path": str(SCRIPT), "generator_sha256": source_hashes[str(SCRIPT)],
        "source_files_sha256": source_hashes, "selection": selection,
        "stored_precision": "Original binary64 values; no rounding performed",
        "is_FV_resume_state": False, "is_nonlinearly_exact_conserved_FV_trajectory": False,
        "limitations": [
            "These are spatially postprocessed point outputs, not a new finite-volume checkpoint or PDE initial state.",
            "Source nonlinear Robin and integral ledgers remain properties of their respective FV trajectories; this point array does not exactly inherit their identities.",
            "Numerical acceptance is supplied by the independently identified analysis, not proven by indexing or output readback.",
            "Preserved historical candidate status records creation before acceptance; the separate accepted analysis identifies the exact candidate bytes.",
            "Physical model assumptions and input uncertainty are not resolved by numerical precision or export rounding."
        ],
        "human_review_claimed": False, "Excel_or_contest_release_created": False}
    metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if args.preflight:
        print(json.dumps({"status": "Q2_AUTHORITY_PREFLIGHT_PASS_NO_FILES_CREATED", "candidate_time_count": candidate_time_count,
                          "selected_integer_time_count": len(indices), "source_files_checked": len(source_hashes)}, ensure_ascii=False))
        return 0
    # Validate all inputs before creating either output. Exclusive file creation
    # prevents overwriting prior results even if another process races this check.
    output.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    progress = {"status": "Q2_AUTHORITY_CREATION_RUNNING_NOT_RELEASED", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "python": sys.executable, "numpy_version": np.__version__,
        "seed": None, "seed_reason": "Deterministic indexing/copying; no random process", "workers": 1,
        "candidate_directory": str(candidate), "source_files_sha256": source_hashes,
        "selection": selection, "output_path": str(output), "metadata": metadata,
        "PDE_solution_performed": False, "final_export_gate_issued": False, "human_review_claimed": False}
    with record.open("x", encoding="utf-8") as log:
        def update_record():
            log.seek(0)
            json.dump(progress, log, ensure_ascii=False, indent=2, allow_nan=False)
            log.truncate()
            log.flush()
            os.fsync(log.fileno())

        update_record()
        try:
            with output.open("xb") as destination:
                np.savez_compressed(destination, **arrays, metadata_json=np.array(metadata_text))
                destination.flush()
                os.fsync(destination.fileno())
            checks = verify_output(output, arrays, metadata_text)
            for path, expected in source_hashes.items():
                verify_hash(Path(path), expected, "source after authority creation")
            progress.update(status="Q2_AUTHORITY_CREATED_EXACT_COPY_VERIFIED", output_sha256=sha256(output),
                            output_bytes=output.stat().st_size, readback=checks,
                            completed_utc=datetime.now(timezone.utc).isoformat())
            update_record()
        except Exception as error:
            progress.update(status="Q2_AUTHORITY_CREATION_FAILED_NOT_RELEASED", error_type=type(error).__name__, error=str(error),
                            output_exists=output.exists(), completed_utc=datetime.now(timezone.utc).isoformat())
            if output.exists():
                progress["unaccepted_output_sha256"] = sha256(output)
            update_record()
            raise
    print(json.dumps({"status": progress["status"], "output": str(output), "output_sha256": progress["output_sha256"],
                      "record": str(record), "PDE_solution_performed": False, "final_export_gate_issued": False,
                      "selected_time_count": 10801, "radius_count": 21}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

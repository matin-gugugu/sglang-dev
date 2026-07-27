#!/usr/bin/env python3
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "experiment-results"
    / "phase0"
    / "schemas"
    / "pattern_demand_v1.schema.json"
)


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    return [Path(path) for path in sorted(set(paths))]


def validate_semantics(record):
    errors = []
    features = record["features"]
    execution = record["execution"]
    histogram = record["group_level_message_histogram"]
    batch_size = features["batch_size"]

    output_lens = features.get("output_lens_per_request")
    if output_lens is not None:
        if len(output_lens) != batch_size:
            errors.append(
                "output_lens_per_request length must equal features.batch_size"
            )
        if output_lens and max(output_lens) != features["output_len"]:
            errors.append("max(output_lens_per_request) must equal features.output_len")

    generated = execution["generated_output_tokens"]
    if output_lens is not None:
        uniform_scalar = (
            len(set(output_lens)) == 1 and generated == features["output_len"]
        )
        if generated != output_lens and not uniform_scalar:
            errors.append(
                "execution.generated_output_tokens must match configured outputs"
            )
    if output_lens is None and generated not in (None, features["output_len"]):
        errors.append("uniform generated_output_tokens must equal features.output_len")

    rebuilt = defaultdict(lambda: {"count": 0, "payload_bytes": 0})
    for index, item in enumerate(histogram):
        prefix = f"group_level_message_histogram[{index}]"
        active_batch_size = item.get("active_batch_size")
        if active_batch_size is not None and active_batch_size > batch_size:
            errors.append(f"{prefix}.active_batch_size exceeds batch_size")

        if item["phase"] == "prefill":
            if (
                item["first_decode_step"] is not None
                or item["last_decode_step"] is not None
            ):
                errors.append(f"{prefix}: prefill record has decode-step bounds")
        else:
            if (
                item.get("prefill_chunk_index") is not None
                or item.get("prefill_chunk_tokens") is not None
            ):
                errors.append(f"{prefix}: decode record has prefill chunk context")
            first = item["first_decode_step"]
            last = item["last_decode_step"]
            if first is None or last is None or first > last:
                errors.append(f"{prefix}: invalid decode-step bounds")

        key = (item["phase"], item["op"])
        rebuilt[key]["count"] += item["count"]
        rebuilt[key]["payload_bytes"] += item["count"] * item["input_payload_bytes"]

    for phase, operations in record["derived"].items():
        for op, values in operations.items():
            expected = rebuilt[(phase, op)]
            if values["count"] != expected["count"]:
                errors.append(f"derived.{phase}.{op}.count does not match histogram")
            if values["payload_bytes"] != expected["payload_bytes"]:
                errors.append(
                    f"derived.{phase}.{op}.payload_bytes does not match histogram"
                )

    for key in rebuilt:
        phase, op = key
        if phase not in record["derived"] or op not in record["derived"][phase]:
            errors.append(f"derived is missing {phase}.{op}")

    validation = record["validation"]
    required_passes = (
        "rank_histogram_consistent",
        "stats_conservation_passed",
        "output_length_consistent",
    )
    for key in required_passes:
        if not validation[key]:
            errors.append(f"validation.{key} is false")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of errors to print before stopping.",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit("no input files")

    schema = json.loads(args.schema.read_text())
    validator = Draft202012Validator(schema)
    row_count = 0
    failures = []

    for path in paths:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row_count += 1
                record = json.loads(line)
                schema_errors = sorted(
                    validator.iter_errors(record), key=lambda error: list(error.path)
                )
                for error in schema_errors:
                    location = ".".join(str(part) for part in error.path) or "<root>"
                    failures.append(
                        f"{path}:{line_number}: schema {location}: {error.message}"
                    )
                for error in validate_semantics(record):
                    failures.append(f"{path}:{line_number}: semantic: {error}")
                if len(failures) >= args.max_errors:
                    break
        if len(failures) >= args.max_errors:
            break

    if failures:
        print("\n".join(failures[: args.max_errors]))
        raise SystemExit(
            f"PatternDemand v1 validation FAILED: "
            f"{len(failures)} error(s), {row_count} row(s) checked"
        )

    print(
        f"PatternDemand v1 validation PASSED: "
        f"{row_count} row(s) across {len(paths)} file(s)"
    )


if __name__ == "__main__":
    main()

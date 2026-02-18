#!/usr/bin/env python3
import os
from pathlib import Path

# ---------------- CONFIG ----------------
USE_ABS_PATH = False  # True = absolute paths, False = relative paths
OUTPUT_FILE = "dataset.py"
# ----------------------------------------


def get_dataset(use_abs_path=True):
    dataset = {}
    idx = 0

    # iterate over all folders in current directory
    for algo_folder in sorted(Path(".").iterdir()):
        if not algo_folder.is_dir():
            continue

        algo_name = algo_folder.name.split("-")[0]  # e.g., RSID, SHA256ID
        hashbits = (
            int(algo_folder.name.split("-")[1]) if "-" in algo_folder.name else 64
        )

        # find the .dlog file (assume 1 per folder)
        dlog_files = list(algo_folder.glob("*.dlog"))
        if not dlog_files:
            continue
        dlog_file = dlog_files[0]

        # gather **all .log files** (no filtering)
        log_files = sorted(algo_folder.glob("*.log"))
        if use_abs_path:
            dlog_path = str(dlog_file.resolve())
            segment_path = str(
                dlog_file.with_name(dlog_file.stem + "_segments_labeled.csv").resolve()
            )
            log_files = [str(f.resolve()) for f in log_files]
        else:
            dlog_path = str(dlog_file)
            segment_path = str(
                dlog_file.with_name(dlog_file.stem + "_segments_labeled.csv")
            )
            log_files = [str(f) for f in log_files]

        dataset[idx] = {
            "algo": algo_name,
            "hashbits": hashbits,
            "segment_csv": segment_path,
            "trace_csv": dlog_path.replace(".dlog", ".csv"),
            "log_files": log_files,
        }
        idx += 1

    return dataset


def save_dataset_py(dataset, output_file):
    with open(output_file, "w") as f:
        f.write("DATASET = {\n")
        for k, v in dataset.items():
            f.write(f"    {k}: {{\n")
            for key, val in v.items():
                if isinstance(val, list):
                    f.write(f"        '{key}': [\n")
                    for item in val:
                        f.write(f"            '{item}',\n")
                    f.write("        ],\n")
                else:
                    f.write(f"        '{key}': '{val}',\n")
            f.write("    },\n")
        f.write("}\n")


if __name__ == "__main__":
    dataset = get_dataset(USE_ABS_PATH)
    save_dataset_py(dataset, OUTPUT_FILE)
    print(f"Saved dataset with {len(dataset)} entries to {OUTPUT_FILE}")

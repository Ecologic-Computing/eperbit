#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
from pprint import pprint
import re
from datetime import datetime
from typing import Dict
import os
import sys
from pathlib import Path
import pandas as pd

# Dataset dict
# from bin.logs.dataset import DATASET
# base_dir = Path("/home/usmanali/eco/dev/usman/idcodes/examples/eperbit/bin/logs")

from DATASET20260212.dataset import DATASET

base_dir = Path(
    "/home/usmanali/eco/dev/usman/idcodes/examples/eperbit/DATASET20260212/"
)
# Prepare CSV header
# csv_header = [
#     "ALGONAME",
#     "ALGOSIZE",
#     "FILENAME",
#     "DATE",
#     "TIME",
#     "STARTTEMP",
#     "ENDTEMP",
#     "INPUTDATATYPE",
#     "VECLENBYTES",
#     "ITERATIONSTOTAL",
#     "AVGTIMEPERITERATIONNSEC",
#     "ITERATIONS_10SEC",
#     "NOTE",
# ]


# Function to parse a log file
def parse_log_file(
    file_path=None,
    algo_name=None,
    algo_size=None,
    base_dir=base_dir,
    append_base_dir=True,
) -> Dict:
    row = {}

    # row["ALGONAME"] = algo_name
    # row["ALGOSIZE"] = algo_size
    log_file_path = Path(file_path)
    if append_base_dir:
        log_file_path = base_dir.joinpath(file_path)
    with open(str(log_file_path), "r") as f:
        log_text = f.read()
    print(f"Parsing log file: {str(log_file_path)}")
    algo_match = re.search(r"(\w+) took on Average", log_text)
    row["ALGONAME"] = algo_match.group(1) if algo_match else ""
    size_match = re.search(r"Option: gf2_exp Value: (\d+)", log_text)
    row["ALGOSIZE"] = size_match.group(1) if size_match else ""
    # Vector length bytes
    vec_len_match = re.search(r"Vector Length \(bytes\): (\d+)", log_text)
    row["VECLENBYTES"] = int(vec_len_match.group(1)) if vec_len_match else ""

    # Iterations
    iter_match = re.search(r"Iter=(\d+)", log_text)
    row["ITERATIONSTOTAL"] = int(iter_match.group(1)) if iter_match else ""

    # Average time per iteration
    avg_time_match = re.search(
        rf"{row['ALGONAME']} took on Average ([\d\.]+) nsec", log_text
    )
    row["AVGTIMEPERITERATIONNSEC"] = (
        float(avg_time_match.group(1)) if avg_time_match else ""
    )
    row["THROUGHPUT_Gbitpersec"] = (
        float(
            ((1 / 1e9) * float(row["VECLENBYTES"]) * 8)
            / (row["AVGTIMEPERITERATIONNSEC"] * 1e-9)
        )
        if row["AVGTIMEPERITERATIONNSEC"] != ""
        else ""
    )
    row["RUNTIME_SEC"] = str(
        f"{(row['ITERATIONSTOTAL'] * row['AVGTIMEPERITERATIONNSEC']) / 1e9:.3f}"
    )
    row["ITERATIONS_10SEC"] = (
        int((10 * 1e9) / row["AVGTIMEPERITERATIONNSEC"])
        if row["AVGTIMEPERITERATIONNSEC"] != ""
        else ""
    )
    row["ITERATIONS_20SEC"] = (
        int((20 * 1e9) / row["AVGTIMEPERITERATIONNSEC"])
        if row["AVGTIMEPERITERATIONNSEC"] != ""
        else ""
    )
    row["TIMEsec_10kITER"] = str(
        f"{(10 * 1e3 * 1e-9) * row['AVGTIMEPERITERATIONNSEC']:.2f}"
    )
    row["MINITERNEEDED_10sOR10k"] = int(max(row["ITERATIONS_10SEC"], 10000))
    row["MINITERNEEDED_20sOR10k"] = int(max(row["ITERATIONS_20SEC"], 10000))

    # Extract START and END temperature
    start_temp_match = re.search(r"System Temperature at START:\s*([\d\.]+)", log_text)
    end_temp_match = re.search(r"System Temperature at END:\s*([\d\.]+)", log_text)
    row["STARTTEMP"] = float(start_temp_match.group(1)) if start_temp_match else ""
    row["ENDTEMP"] = float(end_temp_match.group(1)) if end_temp_match else ""
    row["TEMPCHANGE"] = (
        row["ENDTEMP"] - row["STARTTEMP"]
        if row["ENDTEMP"] != "" and row["STARTTEMP"] != ""
        else ""
    )
    # Input data type
    data_type_match = re.search(r"Option: data_type Value: (\w+)", log_text)
    row["INPUTDATATYPE"] = data_type_match.group(1) if data_type_match else ""

    # Extract START date and time
    start_dt_match = re.search(r"START:::===== ([\d\-]+) ([\d:]+) =====", log_text)
    if start_dt_match:
        row["DATE"] = start_dt_match.group(1)
        row["TIME"] = start_dt_match.group(2)
    else:
        row["DATE"] = ""
        row["TIME"] = ""

    row["FILENAME"] = file_path
    row["NOTE"] = ""

    return row


dfslist = []


def parsedataset(DATASET: dict, base_dir=base_dir):
    rows = []
    master_rows = []
    # Process dataset
    for index, entry in enumerate(DATASET.values()):
        print(
            f"Processing dataset index {index} with algo {entry['algo']} and size {entry['hashbits']}"
        )
        pprint(entry)
        # Use basepath to append to log file paths and get absolute path

        first_log_file = entry["log_files"][0]
        log_dir = os.path.dirname(os.path.abspath(base_dir.joinpath(first_log_file)))
        # CSV will be saved in same dir as first log file
        output_csv = os.path.join(
            log_dir, f"{entry['algo']}-{entry['hashbits']}-summarylogs.csv"
        )

        for log_file in entry["log_files"]:
            # parse_log_file now does not need hardcoded algo/size
            row = parse_log_file(log_file, base_dir=base_dir)
            rows.append(row)

        # Write CSV per dataset entry
        pprint(rows)
        df = pd.DataFrame(rows)
        # Sort by VECLENBYTES (ascending)
        df = df.sort_values(by="VECLENBYTES")
        dfslist.append(df)
        dfcombined = pd.concat(dfslist, ignore_index=True)
        dfcombined.to_csv(
            os.path.join(base_dir, "", "combined_summarylogs.csv"), index=False
        )
        print(
            f"Combined CSV summary written to {os.path.join(base_dir,'', 'combined_summarylogs.csv')}"
        )
        df.to_csv(output_csv, index=False)
        # Create label for this dataset

        label = f"{df['ALGONAME'].iloc[0]}-{df['ALGOSIZE'].iloc[0]}"

        for _, r in df.iterrows():
            master_rows.append(
                {
                    "LABEL": label,
                    "VECLENBYTES": r["VECLENBYTES"],
                    "VALUE": r["MINITERNEEDED_20sOR10k"],
                }
            )
        # print(f"Processed {len(rows)} log files into {output_csv}")
        rows.clear()  # clear for next dataset entry

    master_df = pd.DataFrame(master_rows)

    final_table = master_df.pivot(index="LABEL", columns="VECLENBYTES", values="VALUE")

    # Sort columns numerically
    final_table = final_table.reindex(sorted(final_table.columns), axis=1)

    # Extract algo + size from LABEL index
    index_df = final_table.index.to_series().str.extract(
        r"(?P<ALGONAME>.+)-(?P<SIZE>\d+)"
    )
    index_df["SIZE"] = index_df["SIZE"].astype(int)

    # Sort by ALGONAME then SIZE
    sorted_labels = index_df.sort_values(["ALGONAME", "SIZE"]).index

    final_table = final_table.loc[sorted_labels]

    print(final_table.reset_index())


if __name__ == "__main__":
    # parser = ArgumentParser(description="Parse log files and generate CSV")
    # parser.add_argument(
    #     "output_csv", type=str, help="Path to output CSV file (e.g., output.csv)"
    # )
    # args = parser.parse_args()
    # output_csv = args.output_csv
    # base_dir is path of where logs are located in folder organized manner ALGO-SZ e.g. RSID-8
    parsedataset(DATASET=DATASET, base_dir=base_dir)

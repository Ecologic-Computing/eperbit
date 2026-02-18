#!/usr/bin/env python3
"""
energyperbit.py

Reusable utilities for computing energy, energy per iteration,
and energy per bit from power traces and labeled time segments.
"""

import os
import matplotlib
import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from parselogs import parse_log_file, parsedataset

# We'll target a single-column width figure
fig_width = 3.5 + 1  # inches
fig_height = 2.5 + 1  # inches, adjust for aspect ratio

# Use LaTeX-like fonts for IEEE style
plt.rcParams.update(
    {
        # "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "DeJavu Serif",  # Times font
        # "font.family": "serif",  # Times font
        "font.serif": ["Times"],
        "axes.labelsize": 9,  # Axis labels
        "axes.titlesize": 10,  # Title
        "xtick.labelsize": 8,  # Tick labels
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "figure.figsize": (fig_width, fig_height),
        "figure.dpi": 600,  # High resolution for publication
    }
)
# matplotlib.rc("text", usetex=True)
# params = {"text.latex.preamble": [r"\usepackage{siunitx}", r"\usepackage{cmbright}"]}
# plt.rcParams.update(params)

# Active Core count for each phase (used for accounting power per CPU core)
CORE_MAP = {"warmup": 4, "benchmark": 1}


def read_segments_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a segments CSV robustly:
    - Ignores leading/trailing whitespaces in headers and values
    - Only uses commas as separators
    """
    df = pd.read_csv(
        filepath,
        sep=",",  # only split on commas
        skipinitialspace=True,  # ignores spaces after commas
    )

    # Strip any remaining whitespace from column names
    df.columns = df.columns.str.strip()

    # Optionally strip whitespace from string/object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def parse_bytes(value: str) -> int:
    """
    Convert message size strings to bytes.

    Supported formats:
    - B        (e.g. 32B)
    - KiB/MiB (preferred, binary units)
    - KB/MB   (treated as KiB/MiB for backward compatibility)

    Binary units:
    1 KiB = 1024 bytes
    1 MiB = 1024^2 bytes
    """
    value = value.strip().upper()

    if value.endswith("KIB"):
        return int(float(value[:-3]) * 1024)
    if value.endswith("MIB"):
        return int(float(value[:-3]) * 1024**2)

    # Backward compatibility (treat as binary)
    if value.endswith("KB"):
        return int(float(value[:-2]) * 1024)
    if value.endswith("MB"):
        return int(float(value[:-2]) * 1024**2)

    if value.endswith("B"):
        return int(float(value[:-1]))
    if value.isdigit():
        return int(float(value))

    raise ValueError(f"Unsupported message size format: {value}")


def parse_iterations(value: str) -> int:
    """
    Convert iteration strings (e.g. '10k', '1M') to integer count.
    """
    value = value.strip().lower()

    if value.endswith("k"):
        return int(float(value[:-1]) * 1e3)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1e6)
    if value.isdigit():
        return int(float(value))
    return int(float(value))


def compute_power(
    trace_df: pd.DataFrame,
    voltage_col: str,
    current_col: str,
    power_col: str = "power_W",
) -> pd.DataFrame:
    """
    Adds instantaneous power column to a trace dataframe.
    """
    df = trace_df.copy()
    df[power_col] = df[voltage_col] * df[current_col]
    return df


def compute_average_power(
    trace_df: pd.DataFrame, t_start: float, t_end: float, time_col: str, power_col: str
) -> float:
    """
    Compute average power over a time interval.
    """
    seg = trace_df[(trace_df[time_col] >= t_start) & (trace_df[time_col] <= t_end)]

    if len(seg) < 2:
        return np.nan

    return seg[power_col].mean()


def integrate_corrected_energy(
    trace_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    idle_power_W: float,
    time_col: str,
    power_col: str,
) -> float:
    """
    Integrate power after subtracting idle baseline.
    """
    seg = trace_df[(trace_df[time_col] >= t_start) & (trace_df[time_col] <= t_end)]

    if len(seg) < 2:
        return np.nan

    corrected_power = seg[power_col] - idle_power_W
    corrected_power = corrected_power.clip(lower=0.0)

    return np.trapezoid(corrected_power, seg[time_col])


def compute_phase_aware_energy(
    segments_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    time_col: str,
    power_col: str,
    core_map: dict,
) -> pd.DataFrame:
    """
    Computes:
    - idle average power
    - warmup energy (baseline-subtracted, multi-core)
    - benchmark energy (baseline-subtracted, single-core)

    Only benchmark rows are kept in final output.
    """

    df = segments_df.copy()
    df = df.sort_values("t_start")

    last_idle_power = None
    energies = []

    for _, row in df.iterrows():
        mode = row["mode"]

        if mode == "idle":
            last_idle_power = compute_average_power(
                trace_df, row["t_start"], row["t_end"], time_col, power_col
            )
            energies.append(np.nan)

        elif mode in ("warmup", "benchmark"):
            if last_idle_power is None:
                energies.append(np.nan)
                continue

            energy = integrate_corrected_energy(
                trace_df,
                row["t_start"],
                row["t_end"],
                last_idle_power,
                time_col,
                power_col,
            )

            # Normalize by active cores
            energy /= core_map.get(mode, 1)

            energies.append(energy)

        else:
            energies.append(np.nan)

    df["energy_J"] = energies

    # Only benchmarks go forward
    return df[df["mode"] == "benchmark"].copy()


def compute_baseline_power_between(
    trace_df: pd.DataFrame,
    t_prev_end: float,
    t_curr_start: float,
    time_col: str,
    power_col: str,
) -> float:
    """
    Compute average power between two benchmark phases.
    """
    if t_prev_end >= t_curr_start:
        raise ValueError(
            f"Previous end time {t_prev_end} must be less than current start time {t_curr_start}"
        )

    seg = trace_df[
        (trace_df[time_col] >= t_prev_end) & (trace_df[time_col] <= t_curr_start)
    ]

    if len(seg) < 2:
        return np.nan

    return seg[power_col].mean()


def integrate_energy(
    trace_df: pd.DataFrame, t_start: float, t_end: float, time_col: str, power_col: str
) -> float:
    """
    Integrate power over time using trapezoidal rule.
    """
    segment = trace_df[(trace_df[time_col] >= t_start) & (trace_df[time_col] <= t_end)]

    if len(segment) < 2:
        return np.nan

    return np.trapezoid(segment[power_col], segment[time_col])


def compute_segment_energies(
    segments_df: pd.DataFrame, trace_df: pd.DataFrame, time_col: str, power_col: str
) -> pd.DataFrame:
    """
    Compute total energy for each labeled segment.
    """
    df = segments_df.copy()

    df["energy_J"] = df.apply(
        lambda row: integrate_energy(
            trace_df, row["t_start"], row["t_end"], time_col, power_col
        ),
        axis=1,
    )

    return df


def add_energy_metrics(
    df: pd.DataFrame, hash_bits_map: Dict[str, int], log_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Adds:
    - average power per segment is given by benchmark_power_W
    - latency per iteration
    - energy per iteration
    - energy per cryptographic bit
    - energy per information bit
    """
    out = df.copy()

    # Segment duration
    # out.columns are: Index(['id', 't_start', 't_end', 'mode', 'algo', 'msgsize', 'itertot',
    # #   Column                   Non-Null Count  Dtype
    # ---  ------                   --------------  -----
    # 0   id                       17 non-null     int64
    # 1   t_start                  17 non-null     float64
    # 2   t_end                    17 non-null     float64
    # 3   mode                     17 non-null     object
    # 4   algo                     17 non-null     object
    # 5   msgsize                  17 non-null     object
    # 6   itertot                  17 non-null     object
    # 7   comment                  17 non-null     object
    # 8   msg_bytes                17 non-null     int64
    # 9   iterations               17 non-null     int64
    # 10  benchmark_power_W        17 non-null     float64
    # 11  baseline_power_W         17 non-null     float64
    # 12  energy_J                 17 non-null     float64
    # 13  ALGONAME                 17 non-null     object
    # 14  ALGOSIZE                 17 non-null     object
    # 15  VECLENBYTES              17 non-null     float64
    # 16  ITERATIONSTOTAL          17 non-null     float64
    # 17  AVGTIMEPERITERATIONNSEC  17 non-null     float64
    # 18  THROUGHPUT_Gbitpersec    17 non-null     float64
    # 19  RUNTIME_SEC              17 non-null     object
    # 20  ITERATIONS_10SEC         17 non-null     float64

    out["segment_time_s"] = out["t_end"] - out["t_start"]

    # Average power
    # out["avg_power_W"] = out["energy_J"] / out["segment_time_s"]

    # Latency per iteration
    # out["latency_per_iter_s"] = out["segment_time_s"] / out["iterations"]  # Imprecise
    out["latency_per_iter_s_emeter"] = (
        out["segment_time_s"] / out["iterations"]
    )  # Adding for reference
    out["latency_per_iter_s"] = out["AVGTIMEPERITERATIONNSEC"] / 1e9
    out["hashsize_bits"] = hash_bits_map.get(out["algo"].iloc[0], "Unknown")
    # Energy per iteration
    out["energy_per_iter_J"] = out["energy_J"] / out["iterations"]
    out["E_per_iter_w_baseline_J"] = (
        out["energy_raw_including_baseline_power_J"] / out["iterations"]
    )
    # Cryptographic output bits (e.g. hash length)
    out["hash_bits"] = out["algo"].map(hash_bits_map)
    # Information bits (message size)
    out["info_bits"] = out["msg_bytes"] * 8
    out["energy_per_info_bit_J"] = out["energy_per_iter_J"] / out["info_bits"]
    out["E_per_info_bit_w_baseline_J"] = (
        out["E_per_iter_w_baseline_J"] / out["info_bits"]
    )
    out["tempdelta_C"] = out["TEMPCHANGE"]

    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds human-readable labels like 'SHA-32B'.
    """
    out = df.copy()
    out["label"] = (
        out["algo"]
        + "-"
        + out["hashsize_bits"].astype(str)
        + "-"
        + out["msg_bytes"].astype(str)
        + "B"
    )
    return out


def compute_benchmark_energy_with_interbaseline(
    segments_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    time_col: str,
    power_col: str,
    core_map: dict,
) -> pd.DataFrame:
    """
    Compute baseline-subtracted energy for benchmarks using
    inter-benchmark background power.
    """

    df = segments_df.copy()
    df = df.sort_values("t_start")
    warmup_rows = df[df["mode"] == "warmup"].copy()
    idle_rows = df[df["mode"] == "idle"].copy()
    benchmarks = df[df["mode"] == "benchmark"].copy()
    benchmarks = benchmarks.sort_values("t_start").reset_index(drop=True)

    energies = []
    energies_raw = []
    baseline_powers = []
    benchmark_avg_powers = []
    for i, row in benchmarks.iterrows():
        # if i == 0:
        #     energies.append(np.nan)
        #     baseline_powers.append(np.nan)
        #     continue
        if i == 0:
            prev = warmup_rows.iloc[-1]
        else:
            prev = benchmarks.loc[i - 1]
        # Compute power for idle sections for reference
        baseline_power = compute_baseline_power_between(
            trace_df, prev["t_end"], row["t_start"], time_col, power_col
        )
        benchmark_avg_power = compute_average_power(
            trace_df, row["t_start"], row["t_end"], time_col, power_col
        )
        energy = integrate_corrected_energy(
            trace_df, row["t_start"], row["t_end"], baseline_power, time_col, power_col
        )

        # Normalize by active cores
        energy /= core_map.get("benchmark", 1)
        benchmark_avg_powers.append(benchmark_avg_power)
        energies.append(energy)
        energies_raw.append(
            integrate_energy(
                trace_df, row["t_start"], row["t_end"], time_col, power_col
            )
        )
        baseline_powers.append(baseline_power)
    benchmarks["benchmark_power_W"] = benchmark_avg_powers
    benchmarks["baseline_power_W"] = baseline_powers
    benchmarks["energy_J"] = energies
    benchmarks["energy_raw_including_baseline_power_J"] = energies_raw

    return benchmarks


def run_energy_analysis(
    segment_csv: str,
    trace_csv: str,
    hash_bits_map: Dict[str, int],
    log_df: pd.DataFrame = None,
    time_col: str = "Timestamp [s]",
    voltage_col: str = "N6785A in slot 1 [V]",
    current_col: str = "N6785A in slot 1 [A]",
) -> pd.DataFrame:
    """
    Full energy, power, and latency analysis pipeline.
    """

    # Load CSVs
    # segments = pd.read_csv(segment_csv)
    segments = read_segments_csv(segment_csv)
    trace = pd.read_csv(trace_csv)

    # Parse metadata
    segments["msg_bytes"] = segments["msgsize"].apply(parse_bytes)
    segments["iterations"] = segments["itertot"].apply(parse_iterations)

    # Power computation
    trace = compute_power(trace, voltage_col, current_col)

    # Energy integration
    # Option 1 : no offset of power
    # segments = compute_segment_energies(
    #     segments,
    #     trace,
    #     time_col=time_col,
    #     power_col="power_W"
    # )
    # Option 2: Offset idle power consumption wrong if using spikes in middle as "idle"
    # Phase-aware energy computation
    # segments = compute_phase_aware_energy(
    #     segments,
    #     trace,
    #     time_col=time_col,
    #     power_col="power_W",
    #     core_map=CORE_MAP
    # )
    # Option 3 Offset idle power as power between each benchmark
    segments = compute_benchmark_energy_with_interbaseline(
        segments, trace, time_col=time_col, power_col="power_W", core_map=CORE_MAP
    )
    # TODO
    # Add log_df columns to segments df.: AVGTIMEPERITERATIONNSEC THROUGHPUT_Gbitpersec RUNTIME_SEC STARTTEMP ENDTEMP TEMPCHANGE
    # Ensure that added rows align algo should match ALGONAME and msgsize should match VECLENBYTES itertot should match ITERATIONSTOTAL
    # Add log_df columns to segments df

    cols_to_add = [c for c in log_df.columns if c not in segments.columns]
    for i, row in segments.iterrows():
        match = log_df[
            (log_df["ALGONAME"].astype(str).str.strip() == str(row["algo"]).strip())
            & (log_df["VECLENBYTES"].astype(int) == int(row["msg_bytes"]))
            & (log_df["ITERATIONSTOTAL"].astype(int) == int(row["iterations"]))
        ]
        if not match.empty:
            for col in cols_to_add:
                segments.at[i, col] = match.iloc[0][col]

    # Metrics
    segments = add_energy_metrics(segments, hash_bits_map, log_df)

    # Labels
    segments = add_labels(segments)

    return segments


def generate_msgsize_ticks(max_bytes=1024**3):
    """
    Generate message size ticks and labels in binary units up to max_bytes (default 1GB).

    Returns:
    - tick_values: numeric values in bytes
    - tick_labels: human-readable labels (B, KiB, MiB, GiB)
    """
    units = ["B", "KiB", "MiB", "GiB"]
    tick_values = []
    tick_labels = []

    # Start at 32B, double each step
    val = 32
    while val <= max_bytes:
        tick_values.append(val)
        # Determine appropriate unit
        for i, unit in enumerate(units):
            if val < 1024 ** (i + 1) or unit == "GiB":
                if unit == "B":
                    label = f"{val}B"
                else:
                    # divide by 2**(10*i)
                    label = f"{int(val / (1024**i))}{unit}"
                tick_labels.append(label)
                break
        val *= 2

    return tick_values, tick_labels


def generate_evenly_spaced_msgsize_ticks(max_bytes=1024**3, start_bytes=32):
    """
    Generate evenly spaced ticks for x-axis (visual spacing linear)
    with labels in B/KiB/MiB/GiB.

    Returns:
    - tick_values: positions for x-axis (indices or numeric, equally spaced)
    - tick_labels: human-readable labels
    """
    # Generate powers-of-two message sizes
    sizes = []
    val = start_bytes
    while val <= max_bytes:
        sizes.append(val)
        val *= 2

    # Equally spaced positions
    tick_values = list(range(len(sizes)))  # 0,1,2,... for linear spacing

    # Labels in human-readable format
    tick_labels = []
    units = ["B", "KiB", "MiB", "GiB"]
    for val in sizes:
        for i, unit in enumerate(units):
            if val < 1024 ** (i + 1) or unit == "GiB":
                if unit == "B":
                    label = f"{val}B"
                else:
                    label = f"{int(val / (1024**i))}{unit}"
                tick_labels.append(label)
                break

    return tick_values, tick_labels, sizes  # return sizes for plotting data mapping


def plot_energy_per_bit_multi_algo(
    df,
    energy_col="energy_per_info_bit_J",
    second_col="benchmark_power_W",
    # second_col="latency_per_iter_s",
    algo_col="algo",
    hash_bits_col="hash_bits",
    show_second_col=True,
    figsize=(6, 6),
    dpi=300,
    savepath=None,
):
    """
    Line plot of energy per bit vs message size with evenly spaced x-axis points.
    - Different line styles and markers for different algorithms
    - Energy = blue, Power = orange
    - Legend outside below x-axis
    """
    df_sorted = df.sort_values("msg_bytes").reset_index(drop=True)
    x = np.arange(len(df_sorted))
    unique_msg_bytes = df_sorted["msg_bytes"].unique()
    unique_msg_bytes = np.sort(unique_msg_bytes)  # Ensure sorted order
    unique_msg_sizes = unique_msg_bytes
    msg_to_x_dict = {size: i for i, size in enumerate(unique_msg_sizes)}
    ecologic_colour_palette = {
        "green": "#8fbc85",
        "orange": "#fdb462",
        "darkgreen": "#172613",
    }

    # Human-readable x-axis labels
    def human_readable_bytes(val):
        if val < 1024:
            return f"{val}B"
        elif val < 1024**2:
            return f"{int(val/1024)}KiB"
        elif val < 1024**3:
            return f"{int(val/1024**2)}MiB"
        else:
            return f"{int(val/1024**3)}GiB"

    x_labels = [human_readable_bytes(v) for v in unique_msg_bytes]

    fig, ax_energy = plt.subplots(figsize=figsize, dpi=dpi)
    fig, ax_energy = plt.subplots()
    # Define styles
    line_styles = [
        "solid",
        "--",
        ":",
        "-.",
        (0, (3, 1, 1, 1)),
        (0, (5, 5)),
        (0, (1, 10)),
    ]
    markers = [",", "d", "x", "^", "|", ".", "s", "D", "v", "o", "p", "h"]

    algos = df_sorted[algo_col].unique()
    hash_bits = df_sorted[hash_bits_col].unique()
    combinations = df_sorted[[algo_col, hash_bits_col]].drop_duplicates().values
    style_map = {}
    first_col_to_label_dict = {
        "energy_per_info_bit_J": "Energy/bit",
        "E_per_iter_w_baseline_J": "Energy w/ baseline",
        "E_per_info_bit_w_baseline_J": "Energy/bit w/ baseline",
        "energy_per_iter_J": "Energy",
        "benchmark_power_W": "Avg. Power",
        "latency_per_iter_s": "Latency",
        "E_per_info_bit_w_baseline_J": "Energy/bit w/ baseline",
        "THROUGHPUT_Gbitpersec": "Throughput",
    }
    second_col_to_label_dict = {
        "energy_per_info_bit_J": "Energy/bit",
        "energy_per_iter_J": "Energy",
        "benchmark_power_W": "Avg. Power",
        "latency_per_iter_s": "Latency",
        "E_per_info_bit_w_baseline_J": "Energy/bit w/ baseline",
        "THROUGHPUT_Gbitpersec": "Throughput [Gbit/s]",
    }
    col_to_axis_label_dict = {
        "energy_per_iter_J": "Energy per iteration [\u00b5J]",
        "E_per_iter_w_baseline_J": "Energy per iteration w/ baseline [\u00b5J]",
        "energy_per_info_bit_J": "Energy per info. bit [\u00b5J/bit]",
        "E_per_info_bit_w_baseline_J": "Energy per info. bit w/ baseline [\u00b5J/bit]",
        "THROUGHPUT_Gbitpersec": "Throughput [Gbit/s]",
        "benchmark_power_W": "Avg. Power [W]",
        "latency_per_iter_s": "Latency",
    }
    for i, (algo, h_bits) in enumerate(combinations):
        print(f"Assigning style for {algo} with {h_bits} hash bits")
        ls = line_styles[i % len(line_styles)]
        mk = markers[i % len(markers)]
        # We use a tuple (algo, h_bits) as the dictionary key
        style_map[(algo, h_bits)] = (ls, mk)

    # Plot energy per bit
    for (algo, h_bits), (ls, mk) in style_map.items():
        # Filter for the specific combination
        algo_rows = df_sorted[
            (df_sorted[algo_col] == algo) & (df_sorted[hash_bits_col] == h_bits)
        ].sort_values("msg_bytes")
        # ls, mk = style_map[(algo, algo_rows[hash_bits_col].iloc[0])]
        algo_x = algo_rows["msg_bytes"].map(msg_to_x_dict)
        if "energy" in energy_col or "E_" in energy_col:
            y_val = algo_rows[energy_col] * 1e6
        else:
            y_val = algo_rows[energy_col]
        ax_energy.plot(
            algo_x,
            y_val,  # Convert to microjoules for better visualization
            linestyle=ls,
            marker=mk,
            alpha=0.7,
            color=ecologic_colour_palette["green"],
            linewidth=1.5,
            label=f"{algo}-{algo_rows[hash_bits_col].iloc[0]} { first_col_to_label_dict[energy_col] }",
        )
    if "energy" in energy_col or "E_" in energy_col:
        ax_energy.set_yscale("log")
    ax_energy.set_xlabel("Message size")

    ax_energy.set_ylabel(col_to_axis_label_dict[energy_col])
    # ax_energy.set_ylim(bottom=1e-11, top=0.5e-6)
    # Major ticks every 1e-9
    # ax_energy.yaxis.set_major_locator(ticker.MultipleLocator(0.5e-8))
    # Minor ticks every 0.2e-9
    # ax_energy.yaxis.set_minor_locator(ticker.MultipleLocator(0.2e-9))

    ax_energy.grid(True, linestyle="--", linewidth=0.5, axis="y")

    # Optional secondary axis for power

    if show_second_col:
        ax_second = ax_energy.twinx()
        if second_col == "benchmark_power_W":
            ax_second.set_ylabel("Average power [W]")
            ax_second.set_ylim(bottom=3, top=4.5)

        elif second_col == "latency_per_iter_s":
            ax_second.set_ylabel("Latency per iteration [ms]")
            ax_second.set_ylim(bottom=1e-9, top=5e-2)
            ax_second.set_yscale("log")

    for (algo, h_bits), (ls, mk) in style_map.items():
        # Filter for the specific combination
        algo_rows = df_sorted[
            (df_sorted[algo_col] == algo) & (df_sorted[hash_bits_col] == h_bits)
        ].sort_values("msg_bytes")
        # ls, mk = style_map[(algo, algo_rows[hash_bits_col].iloc[0])]
        algo_x = algo_rows["msg_bytes"].map(msg_to_x_dict)
        if second_col == "latency_per_iter_s":
            # Convert seconds to milliseconds for better visualization
            algo_y = algo_rows[second_col] * 1000
        if second_col == "benchmark_power_W":
            algo_y = algo_rows[second_col]
        else:
            algo_y = algo_rows[second_col]
        ax_second.plot(
            algo_x,
            algo_y,
            linestyle=ls,
            marker=mk,
            color=ecologic_colour_palette["orange"],
            alpha=0.7,
            linewidth=1.0,
            label=f"{algo}-{algo_rows[hash_bits_col].iloc[0]} { second_col_to_label_dict[second_col] }",
        )

    # ax_second.grid(True, linestyle="dotted", linewidth=0.5, axis="y")
    ax_second.tick_params(axis="y", labelsize=8)

    # Build combined legend
    lines1, labels1 = ax_energy.get_legend_handles_labels()
    if show_second_col:
        lines2, labels2 = ax_second.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
    else:
        lines = lines1
        labels = labels1

    # Place legend below x-axis
    ax_energy.legend(
        lines,
        labels,
        # fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
    )
    ax_energy.set_xticks(np.arange(len(unique_msg_bytes)))
    ax_energy.set_xticklabels(
        x_labels,
        rotation=45,
        ha="right",
        #   fontsize=7
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)  # make space for legend

    # if savepath is None:
    savepath = f"eperbitvsmsgsize-{algo}-{algo_rows[hash_bits_col].iloc[0]}_{energy_col}_{second_col}.pdf"
    print(f"Figure saved: {os.path.abspath(savepath)}")
    plt.savefig(savepath, bbox_inches="tight")

    return fig, ax_energy


if __name__ == "__main__":

    from DATASET20260212.dataset import DATASET

    base_path = "./DATASET20260212/"
    # Append base path to all file paths in DATASET

    for idx, entry in DATASET.items():
        entry["segment_csv"] = os.path.join(base_path, entry["segment_csv"])
        entry["trace_csv"] = os.path.join(base_path, entry["trace_csv"])
        entry["log_files"] = [
            os.path.join(base_path, log_file) for log_file in entry["log_files"]
        ]

    for idx, entry in DATASET.items():
        print(idx, entry)

    ii = 0
    dfs = []
    for ind, algo_dataset in enumerate(DATASET.keys()):
        # DATASET[algo_dataset]
        # if DATASET[algo_dataset]["hashbits"] != "64":
        #     print(
        #         "Skipping",
        #         algo_dataset,
        #         DATASET[algo_dataset]["algo"],
        #         DATASET[algo_dataset]["hashbits"],
        #     )
        #     continue
        HASH_BITS = {DATASET[algo_dataset]["algo"]: DATASET[algo_dataset]["hashbits"]}
        segment_csv = DATASET[algo_dataset]["segment_csv"]
        trace_csv = DATASET[algo_dataset]["trace_csv"]
        rows = []
        for log_idx, log_file in enumerate(DATASET[algo_dataset]["log_files"]):
            log_data = parse_log_file(log_file, append_base_dir=False)
            rows.append(log_data)
        log_df = pd.DataFrame(rows)
        df = run_energy_analysis(
            segment_csv=segment_csv,
            trace_csv=trace_csv,
            hash_bits_map=HASH_BITS,
            log_df=log_df,
        )

        dfs.append(df)

        print(
            df[
                [
                    "label",
                    # "msg_bytes",
                    # "hash_bits",
                    "iterations",
                    "benchmark_power_W",
                    "baseline_power_W",
                    "latency_per_iter_s",
                    "energy_per_iter_J",
                    "E_per_iter_w_baseline_J",
                    "THROUGHPUT_Gbitpersec",
                    "energy_per_info_bit_J",
                    "E_per_info_bit_w_baseline_J",
                    "tempdelta_C",
                ]
            ].to_string(max_rows=200, max_cols=None)
        )

    plot_df = pd.concat(dfs, ignore_index=True)
    plot_df.to_csv("finalsummary.csv", index=True)
    # plot_energy_per_bit_multi_algo(
    #     plot_df,
    #     energy_col="energy_per_iter_J",  # E_per_info_bit_w_baseline_J | energy_per_info_bit_J | energy_per_iter_J | E_per_iter_w_baseline_J | THROUGHPUT_Gbitpersec
    #     show_second_col=True,
    #     second_col="benchmark_power_W",  #  "benchmark_power_W" or "latency_per_iter_s"
    #     hash_bits_col="hash_bits",
    #     savepath="energy_per_bit_vs_msgsize.pdf",
    # )

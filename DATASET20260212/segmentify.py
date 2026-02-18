#!/usr/bin/env python3
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd


def loadcsv(csv_path: str) -> pd.DataFrame:
    # Implement your CSV loading logic here
    # For example, you can use pandas to read the CSV file
    # -----------------------------
    # Load data
    # -----------------------------
    # Example CSV load
    # csv_path = "./SHA256ID-64/adg_20260210_135310.csv"
    # df = pd.read_csv(csv_path, skiprows=6) # for dlog skip 6, for scope skip 7
    # df = df.rename(columns={
    #     'volt_avg_1': 'voltage_1',
    #     'curr_avg_1': 'current_1'
    # })
    df = pd.read_csv(csv_path)  # No skip needed if using dlog-viewer
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(df.columns)
    df = df.rename(
        columns={
            "timestamp_[s]": "T",
            "n6785a_in_slot_1_[v]": "V",
            "n6785a_in_slot_1_[a]": "I",
        }
    )
    print(df.columns)

    # Optional temperature column
    if "temperature" not in df.columns:
        df["temperature"] = np.nan

    # -----------------------------
    # Time axis (seconds)
    # -----------------------------
    sampling_interval = 1  # seconds (adjust if known)
    # df["time_s"] = df.index * sampling_interval

    # -----------------------------
    # Derived measurements
    # -----------------------------
    df["P"] = df["V"] * df["I"]
    return df


def segmentify(
    df: pd.DataFrame,
    csv_path: str = "csv_path",
    low_thresh: float = 0.52,
    high_thresh: float = 0.62,
    min_duration: int = 10,
):

    # -----------------------------
    # Statistics
    # -----------------------------
    stats = {
        "mean_voltage": df["V"].mean(),
        "std_voltage": df["V"].std(),
        "mean_current": df["I"].mean(),
        "std_current": df["I"].std(),
        "mean_power": df["P"].mean(),
        "std_power": df["P"].std(),
    }

    # RMS values
    rms_voltage = np.sqrt(np.mean(df["V"] ** 2))
    rms_current = np.sqrt(np.mean(df["I"] ** 2))
    rms_power = np.sqrt(np.mean(df["P"] ** 2))

    # -----------------------------
    # Rolling smoothing
    # -----------------------------
    window = 5
    min_period = 1
    df["voltage_smooth"] = (
        df["V"].rolling(window, min_periods=min_period, center=True).mean()
    )
    df["current_smooth"] = (
        df["I"].rolling(window, min_periods=min_period, center=True).mean()
    )
    df["power_smooth"] = (
        df["P"].rolling(window, min_periods=min_period, center=True).mean()
    )

    # -----------------------------
    # Regression (Power vs Time)
    # -----------------------------
    # X = df["T"]
    # y = df["I"]
    # reg = LinearRegression().fit(X, y)
    # df["power_regression"] = reg.predict(X)

    # -----------------------------
    # Plotting
    # -----------------------------
    # sns.set_theme(style="whitegrid")

    # fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # # Voltage
    # sns.lineplot(ax=axes[0], data=df, x="T", y="V", label="Voltage")
    # sns.lineplot(ax=axes[0], data=df, x="T", y="voltage_smooth", label="Smoothed")
    # axes[0].axhline(stats["mean_voltage"], linestyle="--", label="Mean")
    # axes[0].set_ylabel("Voltage (V)")
    # axes[0].legend()

    # # Current
    # # sns.lineplot(ax=axes[1], data=df, x="time_s", y="current_1", label="Current")
    # sns.lineplot(ax=axes[1], data=df, x="T", y="current_smooth", label="Smoothed")
    # # axes[1].axhline(stats["mean_current"], linestyle="--", label="Mean")
    # axes[1].set_ylabel("Current (A)")
    # axes[1].legend()

    # # Power
    # sns.lineplot(ax=axes[2], data=df, x="T", y="P", label="Power")
    # sns.lineplot(ax=axes[2], data=df, x="T", y="power_smooth", label="Smoothed")
    # # sns.lineplot(ax=axes[2], data=df, x="T", y="power_regression", label="Regression")
    # axes[2].axhline(stats["mean_power"], linestyle="--", label="Mean")
    # axes[2].set_ylabel("Power (W)")
    # axes[2].legend()

    # # Temperature (optional)
    # sns.lineplot(ax=axes[3], data=df, x="T", y="temperature", label="Temperature")
    # axes[3].set_ylabel("Temperature (°C)")
    # axes[3].set_xlabel("Time (s)")
    # axes[3].legend()

    # plt.suptitle(
    #     f"Idle Power Analysis\n"
    #     f"Mean Power: {stats['mean_power']:.4f} W | "
    #     f"RMS Power: {rms_power:.4f} W",
    #     fontsize=14,
    # )

    # plt.tight_layout()
    # # plt.show()
    # # Save figure with same filename as .dlog file appended with type of plot name in png
    # path_csv_file = Path(csv_path)
    # export_plot_path = path_csv_file.with_suffix("").with_name(
    #     f"{path_csv_file.stem}_idle_analysis.png"
    # )
    # plt.savefig(export_plot_path)
    # print(f"Exported idle analysis plot to {export_plot_path}")

    # -----------------------------
    # Print summary
    # -----------------------------
    print("=== Idle Consumption Statistics ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    print("\nRMS Values:")
    print(f"RMS Voltage: {rms_voltage:.6f} V")
    print(f"RMS Current: {rms_current:.6f} A")
    print(f"RMS Power:   {rms_power:.6f} W")

    # low_thresh = 0.62
    # high_thresh = 0.70
    # min_duration = 10  # minimum number of samples to consider a steady state
    state = pd.Series(index=df.index, dtype="object")

    state[df["current_smooth"] <= low_thresh] = "low"
    state[df["current_smooth"] >= high_thresh] = "high"
    state[
        (df["current_smooth"] > low_thresh) & (df["current_smooth"] < high_thresh)
    ] = "mid"
    groups = (state != state.shift()).cumsum()  # group consecutive identical states

    grouped = state.groupby(groups)

    steady_groups = grouped.filter(lambda x: len(x) >= min_duration)
    steady_groups_df = steady_groups.to_frame(name="state")
    steady_groups_df["group"] = groups[steady_groups.index]

    # Get unique groups with their start/end indices and states
    group_info = (
        steady_groups_df.groupby("group")
        .agg(
            start_idx=("state", "idxmin"),
            end_idx=("state", "idxmax"),
            state=("state", "first"),
        )
        .reset_index()
    )

    # Find transitions: low → high (step up), high → low (step down)
    step_events = []

    for i in range(len(group_info) - 1):
        current_state = group_info.loc[i, "state"]
        next_state = group_info.loc[i + 1, "state"]

        if current_state == "low" and next_state == "high":
            # step up event: transition from low to high
            step_up_idx = group_info.loc[i + 1, "start_idx"]  # start of high group
            step_events.append(("up", step_up_idx))

        elif current_state == "high" and next_state == "low":
            # step down event: transition from high to low
            step_down_idx = group_info.loc[i + 1, "start_idx"]  # start of low group
            step_events.append(("down", step_down_idx))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df, x="T", y="current_smooth", ax=ax, label="Smoothed Current")

    for event_type, idx in step_events:
        time = df.loc[idx, "T"]
        current = df.loc[idx, "current_smooth"]
        sample = df.iloc[idx]
        color = "green" if event_type == "up" else "red"
        label = "Step Up" if event_type == "up" else "Step Down"
        marker = "↑" if event_type == "up" else "↓"

        ax.scatter(
            time,
            current,
            color=color,
            s=10,
            label=label if label not in ax.get_legend_handles_labels()[1] else "",
        )
        ax.text(
            time,
            current,
            f"{marker}{sample}",
            color=color,
            fontsize=9,
            ha="center",
            va="bottom" if event_type == "up" else "top",
        )

    ax.set_ylabel("Current (A)")
    ax.legend(loc="upper right")
    # plt.show()
    # Save figure with same filename as .dlog file appended with type of plot name in png
    path_csv_file = Path(csv_path)
    export_plot_path = path_csv_file.with_suffix("").with_name(
        f"{path_csv_file.stem}_step_events.png"
    )
    plt.savefig(export_plot_path)
    print(f"Exported step event plot to {export_plot_path}")

    # Assume df has columns: sample, time_s, voltage_1, current_1 (or smoothed if you prefer)
    # Also assume step_events is a list like [('up', idx1), ('down', idx2), ('up', idx3), ('down', idx4), ...]

    fs = 10  # sampling frequency (samples per second)
    dt = 1 / fs

    # Pair step up and step down events into segments
    segments = []
    waiting_for_down = False
    start_idx = None

    for event_type, idx in step_events:
        if event_type == "up":
            # Start a new segment
            start_idx = idx
            waiting_for_down = True
        elif event_type == "down" and waiting_for_down:
            # End the segment
            end_idx = idx
            segments.append((start_idx, end_idx))
            waiting_for_down = False

    print(f"Found {len(segments)} segments:")
    print(segments)

    # Compute energy for each segment using trapezoidal integration
    # fs = 10  # samples per second
    # dt = 1/fs
    dt = 1
    energies = []
    for start_idx, end_idx in segments:
        segment_df = df.loc[start_idx:end_idx]
        # Use smoothed voltage/current
        power = segment_df["voltage_smooth"] * segment_df["current_smooth"]
        power = power.dropna()
        energy = np.trapezoid(power, dx=dt)
        energies.append(
            {
                "start": segment_df["T"].iloc[0],
                "end": segment_df["T"].iloc[-1],
                "E": energy,
            }
        )

    print("Computed energies:")
    for i, seg in enumerate(energies):
        print(f"Segment {i+1}: {seg['E']:.2f} J, time {seg['start']}s → {seg['end']}s")

    # Now plot energy segments on a new figure aligned with time axis

    fig, ax = plt.subplots(figsize=(14, 4))

    colors = sns.color_palette("tab10", len(energies))
    max_energy = max([e["E"] for e in energies])

    for i, seg in enumerate(energies):
        start_time = seg["start"]
        end_time = seg["end"]
        energy = seg["E"]
        start_idx, end_idx = segments[i]
        # duration = abs(end_idx - start_idx)
        duration = abs(end_time - start_time)
        # Draw bar at bottom with height = energy
        ax.bar(
            x=start_time,
            width=end_time - start_time,
            height=energy,
            bottom=0,
            align="edge",
            color=colors[i],
            alpha=0.6,
        )

        # Annotate bar on top with start/end, duration, and energy
        ax.text(
            x=start_time + (end_time - start_time) / 2,  # center of bar
            y=energy + 0.02 * max_energy,  # slightly above the bar
            s=f"{start_idx}->{end_idx}\nΔ={duration:.1f}ms\nE={energy:.2f} J",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy per Step Segment (Height = Energy, Width = Duration)")
    ax.set_ylim(bottom=0, top=max_energy * 1.2)  # add space above bars for text
    plt.tight_layout()
    # plt.show()
    # Save figure with same filename as .dlog file appended with type of plot name in png
    path_csv_file = Path(csv_path)
    export_plot_path = path_csv_file.with_suffix("").with_name(
        f"{path_csv_file.stem}_energy_segments.png"
    )
    plt.savefig(export_plot_path)
    print(f"Exported energy segment plot to {export_plot_path}")

    # default values (replace these with variables)
    mode_default = "benchmark"
    algo_default = "ALG"
    msgsize_default = "0B"
    itertot_default = "0k"
    comment_default = "none"

    # --- build labeled segments ---
    rows = []
    id = 0
    for start_idx, end_idx in segments:
        t_start = df.loc[start_idx, "T"]
        t_end = df.loc[end_idx, "T"]
        id = id + 1
        rows.append(
            {
                "id": id,
                "t_start": t_start,
                "t_end": t_end,
                "mode": mode_default,
                "algo": algo_default,
                "msgsize": msgsize_default,
                "itertot": itertot_default,
                "comment": comment_default,
            }
        )

    labeled_segments = pd.DataFrame(rows)
    labeled_segments.head()
    # compute durations
    threshold_seconds = 3  # if segment longer than 3 seconds
    durations = (labeled_segments["t_end"] - labeled_segments["t_start"]).abs()
    matches = labeled_segments[durations >= threshold_seconds]

    print(f"Threshold: {threshold_seconds} seconds")
    print(f"Matching segments found: {len(matches)}")

    # update comment column for matching rows
    for idx, row in matches.iterrows():
        duration = abs(row["t_end"] - row["t_start"])
        labeled_segments.at[idx, "comment"] = f"{duration:.3f}s"

        print(f"  id={row['id']}  duration={duration:.6f}s")

    # labeled_segments.to_csv("labeled_segments.csv", index=False)

    # full path + base name without extension
    path_csv_file = Path(csv_path)
    base_no_ext = path_csv_file.with_suffix("")  # removes extension

    orig = path_csv_file
    # final output path
    # out_csv = base_no_ext.with_name(f"{orig.stem}_label_seg.csv")
    export_csv_labelled_segments_path = orig.with_suffix("").with_name(
        f"{orig.stem}_segments_labeled.csv"
    )
    labeled_segments.to_csv(export_csv_labelled_segments_path, index=False)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
    ):

        print(labeled_segments)
    print(f"Exported to {export_csv_labelled_segments_path}")
    return labeled_segments


def preview_matching_segments(labeled_segments: pd.DataFrame, threshold: float = 2):
    """
    Print how many segments meet the duration threshold.
    Updates the 'comment' column with duration (3 decimal places).
    Does NOT write any files.
    """

    # compute durations
    durations = (labeled_segments["t_end"] - labeled_segments["t_start"]).abs()
    matches = labeled_segments[durations >= threshold]

    print(f"Threshold: {threshold} seconds")
    print(f"Matching segments found: {len(matches)}")

    # update comment column for matching rows
    for idx, row in matches.iterrows():
        duration = abs(row["t_end"] - row["t_start"])
        labeled_segments.at[idx, "comment"] = f"{duration:.3f}s"

        print(f"  id={row['id']}  duration={duration:.6f}s")

    return matches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Segmentify current data for states from CSV"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Path to the input CSV file (exported from .dlog)",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.52,
        help="Low threshold for current to define 'low' state",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=0.59,
        help="High threshold for current to define 'high' state",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=5,
        help="Minimum duration (in samples) to consider a steady state",
    )
    args = parser.parse_args()
    if args.csv_path == "":
        print("Please provide a valid CSV file path using --csv_path")
        sys.exit(1)
    csv_path = args.csv_path
    low_thresh = args.low
    high_thresh = args.high
    min_duration = args.min_duration

    # # For quick testing without CLI, you can hardcode values here:
    # csv_path = "./SHA256ID-8/adg_20260210_133347.csv"
    # low_thresh = 0.62
    # high_thresh = 0.70
    # min_duration = 10

    csvf = csv_path
    df = loadcsv(csvf)
    seg_df = segmentify(
        df=df,
        csv_path=csvf,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        min_duration=min_duration,
    )
    print(seg_df)
    matches = preview_matching_segments(seg_df)
    print(matches)

#!/usr/bin/env bash
shopt -s nullglob  # makes globs expand to empty if no match
# Exit on error
set -e
# Find all folders matching the pattern <ALGORITHM>-<CASE>
# Adjust pattern if needed
# folders=()
# for dir in */; do
#     # Remove trailing slash
#     dir="${dir%/}"
#     # Match pattern with dash and number
#     if [[ "$dir" =~ ^[A-Za-z0-9]+-[0-9]+$ ]]; then
#         folders+=("\"$dir\"")
#     fi
# done

# # Print stringified list for copy-paste
# echo "FOLDERS=("
# for f in "${folders[@]}"; do
#     echo "    $f"
# done
# echo ")"

# add flag
REMOVE_EXISTING_SEGMENTS=false  # Set to true to enable removal of existing segment files
# REMOVE_EXISTING_SEGMENTS=true  # Set to true to enable removal of existing segment files

# default
REMOVE_EXISTING_SEGMENTS=false

# if user passed an argument, override
if [[ $# -ge 1 ]]; then
    case "$1" in
        true|false)
            REMOVE_EXISTING_SEGMENTS="$1"
            ;;
        *)
            echo "Error: expected 'true' or 'false' but got '$1'"
            exit 1
            ;;
    esac
fi

echo "REMOVE_EXISTING_SEGMENTS=$REMOVE_EXISTING_SEGMENTS"



# List of folders to process (modify as needed)
FOLDERS=(
    "PMHID-16"
    "PMHID-32"
    "PMHID-64"
    "PMHID-8"
    "RMID-16"
    "RMID-32"
    "RMID-64"
    "RMID-8"
    "RS2ID-16"
    "RS2ID-32"
    "RS2ID-64"
    "RSID-16"
    "RSID-32"
    "RSID-64"
    "RSID-8"
    "SHA256ID-16"
    "SHA256ID-32"
    "SHA256ID-64"
    "SHA256ID-8"
)

for folder in "${FOLDERS[@]}"; do
    echo "Processing folder: $folder"

    # Only select CSV files that do NOT end with _segments_labeled.csv
    for csv_file in "$folder"/*.csv; do
        [[ -f "$csv_file" ]] || continue
        [[ "$csv_file" != *_segments_labeled.csv ]] || continue

        # Determine the target segment CSV filename
        segment_csv="${csv_file%.csv}_segments_labeled.csv"

        # Skip if segment CSV already exists and is non-empty
        if [[ -s "$segment_csv" ]]; then
            
            # Add conditional remove if flag enabled (e.g., REMOVE_EXISTING_SEGMENTS=true)
            if (( $REMOVE_EXISTING_SEGMENTS=="true")); then
                echo "Removing $segment_csv (segment file already exists)"
                echo "Removing $segment_csv (flag enabled) REMOVE_EXISTING_SEGMENTS=$REMOVE_EXISTING_SEGMENTS"
                rm "$segment_csv"
                continue
            else
            echo "Skipping $csv_file (segment file already exists)"
            continue
            fi
        fi

        # Run segmentify
        echo "Running segmentify on $csv_file"
        ./segmentify.py --csv_path "$csv_file"

        # Confirm creation
        if [[ -f "$segment_csv" ]]; then
            echo "Created $segment_csv"
        else
            echo "Warning: $segment_csv not created!"
        fi
    done
done

#!/usr/bin/env bash

# Exit on error
set -e

# Loop through all .dlog files recursively
find . -type f -name "*.dlog" | while read -r dlog_file; do
    # Get the directory and base filename
    dir=$(dirname "$dlog_file")
    base=$(basename "$dlog_file" .dlog)
    
    # CSV output path
    csv_file="$dir/$base.csv"
    
    echo "Converting $dlog_file -> $csv_file"
    
    # Run dlog-viewer CLI
    ./dlogviewer.py "$dlog_file" --csv-export "$csv_file"
done

echo "All .dlog files converted to .csv"

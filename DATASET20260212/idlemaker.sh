#!/usr/bin/env bash

csv="$1"

if [[ ! -f "$csv" ]]; then
    echo "File not found: $csv"
    exit 1
fi

# Create backup
cp "$csv" "$csv.bkp"

# Process CSV
awk -F',' '
BEGIN { OFS="," }

NR==1 { print; next }

# First non-none comment → warmup
$8 != "none" && !warmup_done {
    $4 = "warmup"
    warmup_done = 1
    print
    next
}

# comment == none → idle
$8 == "none" {
    $4 = "idle"
    print
    next
}

# Everything else unchanged
{ print }
' "$csv.bkp" > "$csv"

#!/usr/bin/env bash

shopt -s nullglob
DATE="20260212"

for file in *-"$DATE"_*.log; do
    # Split on the first three '-' only
    IFS='-' read -r algorithm case size rest <<< "$file"

    # CASE is the value right after the first '-'
    case "$case" in
        8|16|32|64) ;;
        *) continue ;;
    esac

    dir="${algorithm}-${case}"

    mkdir -p "$dir"
    mv "$file" "$dir/"
done

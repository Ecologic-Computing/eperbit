find . -type f -name "*.dlog" | while read -r dlog; do
    csv="${dlog%.dlog}.csv"

    if [[ -f "$csv" ]]; then
        echo "FOUND: $csv"
        # read from the terminal, not from the pipe
        read -r -p "Delete this file? Type 'y' or 'yes' to confirm: " ans </dev/tty

        case "$ans" in
            y|Y|yes|YES|Yes)
                rm -v -- "$csv"
                ;;
            *)
                echo "Skipped: $csv"
                ;;
        esac
    fi
done

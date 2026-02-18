#!/usr/bin/env bash

# Predefined target directory
TARGET_DIR="/home/rpi/eco/idcodes/out/build/GCC_aarch64_RELEASE_RPI5/bin"

# Predefined list of script base names (without .sh)
SCRIPTS=(
    scope
    test
    screen_test
    warmup
)

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"
# mkdir -p "$TARGET_DIR/logs"
ln -sf "$(pwd)/logs" "$TARGET_DIR/logs"
# Create symlinks
for script in "${SCRIPTS[@]}"; do
    ln -sf "$(pwd)/${script}.sh" "$TARGET_DIR/${script}.sh"
done

echo "Symlinks created in $TARGET_DIR"

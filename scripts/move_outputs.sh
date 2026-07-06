#!/bin/bash

# move_outputs.sh - Move all outputs from a source folder to a destination folder
# ./move_outputs.sh /home/sotgab/sysops/HIPPA/GAB/full_preprocessing_pipeline/outputs/SWIR /home/sotgab/sysops/HIPPA/bruising/processed/SWIR
# ./move_outputs.sh /home/sotgab/sysops/HIPPA/GAB/full_preprocessing_pipeline/outputs/VNIR /home/sotgab/sysops/HIPPA/bruising/processed/VNIR

SOURCE="$1"
DEST="$2"

# --- Validation ---
if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
  echo "Usage: $0 <source_folder> <destination_folder>"
  exit 1
fi

if [ ! -d "$SOURCE" ]; then
  echo "Error: Source folder '$SOURCE' does not exist."
  exit 1
fi

# Create destination folder if it doesn't exist
mkdir -p "$DEST"

# --- Move files ---
COUNT=0
for FILE in "$SOURCE"/*; do
  if [ -f "$FILE" ]; then
    mv "$FILE" "$DEST/"
    echo "Moved: $(basename "$FILE")"
    COUNT=$((COUNT + 1))
  fi
done

echo ""
echo "Done! $COUNT file(s) moved from '$SOURCE' to '$DEST'."
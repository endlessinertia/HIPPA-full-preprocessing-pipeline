#!/bin/bash
 
# delete_matching.sh - Delete files in TARGET folder that match filenames in REFERENCE folder
 
REFERENCE="$1"
TARGET="$2"
 
# --- Validation ---
if [ -z "$REFERENCE" ] || [ -z "$TARGET" ]; then
  echo "Usage: $0 <reference_folder> <target_folder>"
  exit 1
fi
 
if [ ! -d "$REFERENCE" ]; then
  echo "Error: Reference folder '$REFERENCE' does not exist."
  exit 1
fi
 
if [ ! -d "$TARGET" ]; then
  echo "Error: Target folder '$TARGET' does not exist."
  exit 1
fi
 
# --- Delete matching files ---
COUNT=0
NOT_FOUND=0
 
for FILE in "$REFERENCE"/*; do
  if [ -f "$FILE" ]; then
    FILENAME=$(basename "$FILE")
    TARGET_FILE="$TARGET/$FILENAME"
 
    if [ -f "$TARGET_FILE" ]; then
      rm "$TARGET_FILE"
      echo "Deleted: $FILENAME"
      COUNT=$((COUNT + 1))
    else
      echo "Not found, skipped: $FILENAME"
      NOT_FOUND=$((NOT_FOUND + 1))
    fi
  fi
done
 
echo ""
echo "Done! $COUNT file(s) deleted, $NOT_FOUND not found/skipped."
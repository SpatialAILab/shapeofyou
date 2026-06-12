#!/bin/bash
IMAGE_DIR="../../../data/SPair-71k/JPEGImages"
SAVE_DIR="../../../data/SPair-71k"
PAIR_DIR="../../../data/SPair-71k/PairAnnotation"

for mode in trn val test; do
  echo "Running mode: $mode"
  for category in "$IMAGE_DIR"/*; do
    if [ -d "$category" ]; then
      category_name=$(basename "$category")
      echo "Processing category: $category_name in mode: $mode"

      python extract_point.py \
        --dataset_name SPair-71k \
        --image_dir "$IMAGE_DIR" \
        --save_dir "$SAVE_DIR" \
        --category "$category_name" \
        --pair_dir "$PAIR_DIR" \
        --mode "$mode" \
        --num_threads 4
    fi
  done
done
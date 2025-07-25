#!/bin/bash

set -e

PYTHON_SCRIPT="finetuning.py"

#CONFIGURATIONS=(
#  "--test_mode zero --model gemma2 --examples_path examples.tsv"
#  "--test_mode few --model gemma2 --examples_path examples.tsv"
#  "--test_mode def --model gemma2 --examples_path examples.tsv"
#  "--test_mode def-few --model gemma2 --examples_path examples.tsv"
#  "--test_mode zero --model gemma3_1b --examples_path examples.tsv"
#  "--test_mode few --model gemma3_1b --examples_path examples.tsv"
#  "--test_mode def --model gemma3_1b --examples_path examples.tsv"
#  "--test_mode def-few --model gemma3_1b --examples_path examples.tsv"
#  "--test_mode zero --model gemma2 --examples_path context_examples.tsv"
#  "--test_mode few --model gemma2 --examples_path context_examples.tsv"
#  "--test_mode def --model gemma2 --examples_path context_examples.tsv"
#  "--test_mode def-few --model gemma2 --examples_path context_examples.tsv"
#  "--test_mode zero --model gemma3_1b --examples_path context_examples.tsv"
#  "--test_mode few --model gemma3_1b --examples_path context_examples.tsv"
#  "--test_mode def --model gemma3_1b --examples_path context_examples.tsv"
#  "--test_mode def-few --model gemma3_1b --examples_path context_examples.tsv"
#)

# CONFIGURATIONS=(
#   "--test_mode def --model gemma2 --examples_path examples.csv"
#   "--test_mode def-few --model gemma2 --examples_path examples.csv"
#   "--test_mode def --model gemma3_1b --examples_path examples.csv"
#   "--test_mode def-few --model gemma3_1b --examples_path examples.csv"
#   "--test_mode def --model gemma2 --examples_path context_examples.csv"
#   "--test_mode def-few --model gemma2 --examples_path context_examples.csv"
#   "--test_mode def --model gemma3_1b --examples_path context_examples.csv"
#   "--test_mode def-few --model gemma3_1b --examples_path context_examples.csv"
# )

CONFIGURATIONS=(
 "--test_mode zero --model gemma2"
 "--test_mode zero --model gemma2 --training"
 "--test_mode few --model gemma2"
 "--test_mode def --model gemma2"
 "--test_mode def-few --model gemma2"
 "--test_mode zero --model gemma3_1b"
 "--test_mode zero --model gemma3_1b --training"
 "--test_mode few --model gemma3_1b"
 "--test_mode def --model gemma3_1b"
 "--test_mode def-few --model gemma3_1b"
 "--test_mode zero --model gemma3n_e2b_it"
 "--test_mode zero --model gemma3n_e2b_it --training"
 "--test_mode few --model gemma3n_e2b_it"
 "--test_mode def --model gemma3n_e2b_it"
 "--test_mode def-few --model gemma3n_e2b_it"
)


#rm accuracy_example_pool_sizes.csv

for CONFIG in "${CONFIGURATIONS[@]}"; do
  echo "Running configuration: $CONFIG"

  python $PYTHON_SCRIPT $CONFIG

  echo "Completed configuration: $CONFIG"
  echo "----------------------------------------"
done

echo "All configurations tested successfully!"
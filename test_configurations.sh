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
#  "--test_mode zero --model gemma2"
#  "--test_mode zero --model gemma2 --training"
#  "--test_mode few --model gemma2"
#  "--test_mode def --model gemma2"
#  "--test_mode def-few --model gemma2"
#  "--test_mode zero --model gemma3_1b"
#  "--test_mode zero --model gemma3_1b --training"
#  "--test_mode few --model gemma3_1b"
#  "--test_mode def --model gemma3_1b"
#  "--test_mode def-few --model gemma3_1b"
#  "--test_mode zero --model gemma3n_e2b_it"
#  "--test_mode zero --model gemma3n_e2b_it --training"
#  "--test_mode few --model gemma3n_e2b_it"
#  "--test_mode def --model gemma3n_e2b_it"
#  "--test_mode def-few --model gemma3n_e2b_it"

#  "--test_mode zero --model gemma2"
#  "--test_mode zero --model gemma3_1b"
#  "--test_mode zero --model gemma3n_e2b_it"

#  "--test_mode few --model gemma2"
#  "--test_mode few --model gemma3_1b"
#  "--test_mode few --model gemma3n_e2b_it"

#  "--test_mode def --model gemma2"
#  "--test_mode def --model gemma3_1b"
#  "--test_mode def --model gemma3n_e2b_it"

#  "--test_mode def-few --model gemma2"
#  "--test_mode def-few --model gemma3_1b"
#  "--test_mode def-few --model gemma3n_e2b_it"

# "--model gemma2 --test_mode paraph"
# "--model gemma3_1b --test_mode paraph"
# "--model gemma3n_e2b_it --test_mode paraph"

# "--model gemma3_1b --training_examples_experiment --use_early_stopping"
# "--model gemma2 --training_examples_experiment --use_early_stopping"
# "--model gemma3n_e2b_it --training_examples_experiment --use_early_stopping"
# "--model gemma3_1b --training_examples_experiment --use_early_stopping"
# "--model gemma2 --training_examples_experiment --use_early_stopping"
# "--model gemma3n_e2b_it --training_examples_experiment --use_early_stopping"

# FOR ACCURACY COMPARISON GRAPHS
# "--test_mode few --model gemma3_1b"
# "--test_mode def --model gemma3_1b"
# "--test_mode def-few --model gemma3_1b"
# "--test_mode few --model gemma2"
"--test_mode def --model gemma2"
# "--test_mode def-few --model gemma2"
# "--test_mode few --model gemma3n_e2b_it"
# "--test_mode def --model gemma3n_e2b_it"
# "--test_mode def-few --model gemma3n_e2b_it"
# "--model gemma3_1b --test_mode zero"
# "--model gemma3_1b --test_mode paraph"
# "--model gemma3_1b --test_mode zero --training --use_early_stopping"
# "--model gemma3_1b --test_mode paraph --training --use_early_stopping"
# "--model gemma2 --test_mode zero"
# "--model gemma2 --test_mode paraph"
# "--model gemma2 --test_mode zero --training --use_early_stopping"
# "--model gemma2 --test_mode paraph --training --use_early_stopping"
# "--model gemma3n_e2b_it --test_mode zero"
# "--model gemma3n_e2b_it --test_mode paraph"
# "--model gemma3n_e2b_it --test_mode zero --training --use_early_stopping"
# "--model gemma3n_e2b_it --test_mode paraph --training --use_early_stopping"
# "--model gemma3n_e2b_it --test_mode paraph --training --use_early_stopping"

# # FOR TRAINING DATASET SIZE COMPARISON GRAPHS
# "--model gemma3_1b --test_mode zero"
# "--model gemma3_1b --training --examples_path examples.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_180.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_240.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_300.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_360.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --test_mode zero"
# "--model gemma2 --training --examples_path examples.tsv --test_mode zero --use_early_stopping"
# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_180.tsv --test_mode zero --use_early_stopping"
# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_240.tsv --test_mode zero --use_early_stopping"
# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_300.tsv --test_mode zero --use_early_stopping"
# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_360.tsv --test_mode zero --use_early_stopping"

# "--model gemma3n_e2b_it --test_mode zero"
# "--model gemma3n_e2b_it --training --examples_path examples.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_180.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_240.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_300.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_360.tsv --test_mode zero --use_early_stopping"

# FOR TRAINING DATASET SIZE COMPARISON GRAPHS WITH EARLY STOPPING, to be used with force new training
# "--model gemma2 --training --examples_path examples.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path examples.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path examples.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_216.tsv --test_mode zero --use_early_stopping"
# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_324.tsv --test_mode zero --use_early_stopping"
# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_432.tsv --test_mode zero --use_early_stopping"

# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_216.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_324.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_432.tsv --test_mode zero --use_early_stopping"

# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_216.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_324.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_432.tsv --test_mode zero --use_early_stopping"


#EPOCH EXPERIMENTS
# "--model gemma2 --epoch_experiment"
# "--model gemma3_1b --epoch_experiment"


# "--model gemma2 --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_216.tsv"

# "--model gemma3_1b --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_216.tsv"

# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_216.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_216.tsv"

# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_216.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_216.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_216.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_324.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_324.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_324.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --training --examples_path ./dataset/training/training_dataset_432.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/training/training_dataset_432.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/training/training_dataset_432.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_324.tsv"

# "--model gemma3_1b --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_324.tsv"

# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_324.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_324.tsv"

# "--model gemma2 --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma2 --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_432.tsv"

# "--model gemma3_1b --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3_1b --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_432.tsv"

# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 1 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 2 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 3 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 4 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 5 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 6 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 7 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 8 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 9 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 10 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 11 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 12 --examples_path ./dataset/training/training_dataset_432.tsv"
# "--model gemma3n_e2b_it --epoch_experiment --epoch_number_max 13 --examples_path ./dataset/training/training_dataset_432.tsv"


#data augmentation 
# "--paraphrase_and_extend 1 --examples_path ./dataset/alternative_dataset/news_training_dataset.tsv"
# "--paraphrase_and_extend 2 --examples_path ./dataset/alternative_dataset/news_training_dataset.tsv"
# "--paraphrase_and_extend 3 --examples_path ./dataset/alternative_dataset/news_training_dataset.tsv"
# "--paraphrase_and_extend 4"
# "--paraphrase_and_extend 5"

#ALTERNATIVE DATASET TESTING

# "--test_mode zero --model gemma2"
# "--test_mode zero --model gemma3_1b"
# "--test_mode zero --model gemma3n_e2b_it"

# "--test_mode few --model gemma2"
# "--test_mode few --model gemma3_1b"
# "--test_mode few --model gemma3n_e2b_it"

# "--test_mode def --model gemma2"
# "--test_mode def --model gemma3_1b"
# "--test_mode def --model gemma3n_e2b_it"

# "--test_mode def-few --model gemma2"
# "--test_mode def-few --model gemma3_1b"
# "--test_mode def-few --model gemma3n_e2b_it"

# "--model gemma2 --test_mode paraph"
# "--model gemma3_1b --test_mode paraph"
# "--model gemma3n_e2b_it --test_mode paraph"

# "--model gemma2 --training --examples_path ./dataset/alternative_dataset/news_training_dataset.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/alternative_dataset/news_training_dataset.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/alternative_dataset/news_training_dataset.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_116.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_116.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_116.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_174.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_174.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_174.tsv --test_mode zero --use_early_stopping"

# "--model gemma2 --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_232.tsv --test_mode zero --use_early_stopping"
# "--model gemma3_1b --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_232.tsv --test_mode zero --use_early_stopping"
# "--model gemma3n_e2b_it --training --examples_path ./dataset/alternative_dataset/extended_training_datasets/training_dataset_232.tsv --test_mode zero --use_early_stopping"

)


#rm accuracy_example_pool_sizes.csv

for CONFIG in "${CONFIGURATIONS[@]}"; do
  echo "Running configuration: $CONFIG"

  python $PYTHON_SCRIPT $CONFIG

  echo "Completed configuration: $CONFIG"
  echo "----------------------------------------"
done

echo "All configurations tested successfully!"
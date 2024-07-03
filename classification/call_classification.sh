#!/usr/bin/env bash

# Loop over batch numbers
# for batch_number in 1 2; do
for batch_number in 2; do
    # Submit the classification job
    qsub -v batch_number=$batch_number,classification_type=averaged run_classification.pbs
done
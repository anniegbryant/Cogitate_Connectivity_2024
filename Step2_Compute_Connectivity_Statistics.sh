#!/usr/bin/env bash

# Update to where your bids_root directory is
bids_root=/project/hctsa/annie/data/

# Define derivatives directories
export coreg_dir=${bids_root}/derivatives/coreg
export out_fw=${bids_root}/derivatives/forward

# Batch 1 subject list
subject_list_batch_1=subject_list_batch1.txt

# Batch 2 subject list
subject_list_batch_2=subject_list_batch2.txt

# Using visit 1
visit=1

# Using record run
record=run

######################################################################################
# Compute pyspi statistics (fast subset) for all participants
######################################################################################

# Use 1 job by default, you can increase as your system allows
n_jobs=1

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Computing SPIs for $subject"
    # Run pyspi computations for subject averaged epochs
    python3 feature_extraction/run_pyspi_for_subject_averaged_epochs.py --sub $subject --visit_id $visit \
    --bids_root $bids_root/Cogitate_Batch1/MEG_Data --region_option hypothesis_driven --n_jobs $n_jobs \
    --duration "1000ms"
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Computing SPIs for $subject"
    # Run pyspi computations for subject averaged epochs
    python3 feature_extraction/run_pyspi_for_subject_averaged_epochs.py --sub $subject --visit_id $visit \
    --bids_root $bids_root/Cogitate_Batch2/MEG_Data --region_option hypothesis_driven --n_jobs $n_jobs \
    --duration "1000ms"
done
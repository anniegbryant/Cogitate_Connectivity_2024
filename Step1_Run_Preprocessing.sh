#!/usr/bin/env bash

# Update to where your bids_root directory is
bids_root=/project/hctsa/annie/data/

# Replace with your FreeSurfer installation locations
module load freesurfer/7.1.1
export FREESURFER_HOME=/usr/local/freesurfer/7.1.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=${bids_root}/derivatives/fs
export PATH="$FREESURFER_HOME/bin:$PATH"
export PATH="$FREESURFER_HOME/fsfast/bin:$PATH"

# Define derivatives directories
export coreg_dir=${bids_root}/derivatives/coreg
export out_fw=${bids_root}/derivatives/forward

# Update to where your cogitate-msp1 github repo directory is
MEG_repo_root=/project/hctsa/annie/github/cogitate-msp1/coglib

# Batch 1 subject list
subject_list_batch_1=subject_list_batch1.txt

# Batch 2 subject list
subject_list_batch_2=subject_list_batch2.txt

# Using visit 1
visit=1

# Using record run
record=run

######################################################################################
# MNE preprocessing
######################################################################################

# Iterate over subjects in the given data directory
echo "Running MNE preprocessing"

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Processing subject $subject"
    # Run the preprocessing script for step 1
    python3 $MEG_repo_root/meeg/preprocessing/P99_run_preproc.py \
    --sub $subject --visit $visit --record $record --step 1 \
    --bids_root $bids_root/Cogitate_Batch1/MEG_Data

    # Run the preprocessing script for step 2
    python3 $MEG_repo_root/meeg/preprocessing/P99_run_preproc.py \
    --sub $subject --visit $visit --record $record --step 2 \
    --bids_root $bids_root/Cogitate_Batch2/MEG_Data
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Processing subject $subject"
    # Run the preprocessing script for step 1
    python3 $MEG_repo_root/meeg/preprocessing/P99_run_preproc.py \
    --sub $subject --visit $visit --record $record --step 1 \
    --bids_root $bids_root/Cogitate_Batch1/MEG_Data

    # Run the preprocessing script for step 2
    python3 $MEG_repo_root/meeg/preprocessing/P99_run_preproc.py \
    --sub $subject --visit $visit --record $record --step 2 \
    --bids_root $bids_root/Cogitate_Batch2/MEG_Data
done

######################################################################################
# Run FreeSurfer's recon-all pipeline
######################################################################################

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Running recon-all for subject $subject"
    # Run the recon-all script
    recon-all -all -subjid sub-${subject} -i ${bids_root}/Cogitate_Batch1/MEG_Data/sub-${subject}/ses-1/anat/sub-${subject}_ses-1_T1w.nii.gz -sd ${bids_root}/Cogitate_Batch1/MEG_Data/derivatives/fs
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Running recon-all for subject $subject"
    # Run the recon-all script
    recon-all -all -subjid sub-${subject} -i ${bids_root}/Cogitate_Batch2/MEG_Data/sub-${subject}/ses-1/anat/sub-${subject}_ses-1_T1w.nii.gz -sd ${bids_root}/Cogitate_Batch2/MEG_Data/derivatives/fs
done

######################################################################################
# Run MNE's scalp reconstruction
######################################################################################

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Running MNE's scalp reconstruction for subject $subject"
    # Run the scalp reconstruction script
    python3 $MEG_repo_root/meeg/source_modelling/S00a_scalp_surfaces.py \
    --sub $subject --visit $visit --bids_root $bids_root/Cogitate_Batch1/MEG_Data --fs_home $FREESURFER_HOME --subjects_dir $SUBJECTS_DIR
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Running MNE's scalp reconstruction for subject $subject"
    # Run the scalp reconstruction script
    python3 $MEG_repo_root/meeg/source_modelling/S00a_scalp_surfaces.py \
    --sub $subject --visit $visit --bids_root $bids_root/Cogitate_Batch2/MEG_Data --fs_home $FREESURFER_HOME --subjects_dir $SUBJECTS_DIR
done

######################################################################################
# Fit Single-shell Boundary Elements Model
######################################################################################

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Fitting Single-shell Boundary Elements Model for subject $subject"

    # Run the BEM script
    python3 $MEG_repo_root/meeg/source_modelling/S00b_bem.py \
    --sub $subject --visit $visit --bids_root $bids_root/Cogitate_Batch1/MEG_Data --fs_home $FREESURFER_HOME --subjects_dir $SUBJECTS_DIR

    # Run forward model
    python3 $MEG_repo_root/meeg/source_modelling/S01_forward_model.py \
    --sub $subject --visit $visit --space surface --bids_root $bids_root/Cogitate_Batch1/MEG_Data --subjects_dir $SUBJECTS_DIR \
    --coreg_path $coreg_dir --out_fw $out_fw
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Fitting Single-shell Boundary Elements Model for subject $subject"

    # Run the BEM script
    python3 $MEG_repo_root/meeg/source_modelling/S00b_bem.py \
    --sub $subject --visit $visit --bids_root $bids_root/Cogitate_Batch2/MEG_Data --fs_home $FREESURFER_HOME --subjects_dir $SUBJECTS_DIR

    # Run forward model
    python3 $MEG_repo_root/meeg/source_modelling/S01_forward_model.py \
    --sub $subject --visit $visit --space surface --bids_root $bids_root/Cogitate_Batch2/MEG_Data --subjects_dir $SUBJECTS_DIR \
    --coreg_path $coreg_dir --out_fw $out_fw
done

######################################################################################
# Run subject-level source localization
######################################################################################

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Running subject-level source localization for subject $subject"

    # Run the source localization script using dspm
    python3 $MEG_repo_root/meeg/activation/S01_source_loc.py \
    --sub $subject --visit $visit --method dspm \
    --bids_root $bids_root/Cogitate_Batch1/MEG_Data
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Running subject-level source localization for subject $subject"

    # Run the source localization script using dspm
    python3 $MEG_repo_root/meeg/activation/S01_source_loc.py \
    --sub $subject --visit $visit --method dspm \
    --bids_root $bids_root/Cogitate_Batch2/MEG_Data
done

######################################################################################
# Run global source localization
######################################################################################

echo "Running global source localization for Batch 1"
# Run the global source localization script using dspm
# Iterate over the alpha, beta, and gamma bands
for band in alpha beta gamma; do
    python3 $MEG_repo_root/meeg/activation/S02_source_loc_ga.py --visit $visit --band $band --bids_root $bids_root/Cogitate_Batch1/MEG_Data \
    --method dspm --participants_file_list subject_list_batch1.txt
done

echo "Running global source localization for Batch 2"
# Run the global source localization script using dspm
# Iterate over the alpha, beta, and gamma bands
for band in alpha beta gamma; do
    python3 $MEG_repo_root/meeg/activation/S02_source_loc_ga.py --visit $visit --band $band --bids_root $bids_root/Cogitate_Batch2/MEG_Data \
    --method dspm --participants_file_list subject_list_batch2.txt
done

######################################################################################
# Average across epochs to get a single event-related field (ERF) time series per meta-ROI
######################################################################################

# Use 1 job by default, you can increase as your system allows
n_jobs=1

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Averaging across ERFs for subject $subject"

    # Run the script to average across epochs
    python3 MEG_preprocessing/extract_time_series_from_MEG.py --sub $subject \
    --bids_root $bids_root/Cogitate_Batch1/MEG_Data --n_jobs $n_jobs --region_option hypothesis_driven
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Averaging across ERFs for subject $subject"

    # Run the script to average across epochs
    python3 MEG_preprocessing/extract_time_series_from_MEG.py --sub $subject \
    --bids_root $bids_root/Cogitate_Batch2/MEG_Data --n_jobs $n_jobs --region_option hypothesis_driven
done

######################################################################################
# Combine time series across participants into CSV files
######################################################################################

# Batch 1
echo "Combining time series across participants"

# Batch 1
for subject in $(cat $subject_list_batch_1); do
    echo "Averaging across ERFs for subject $subject"

    # Run the script to average across epochs
    python3 MEG_preprocessing/combine_time_series_from_MEG.py --sub $subject \
    --bids_root $bids_root/Cogitate_Batch1/MEG_Data --region_option hypothesis_driven
done

# Batch 2
for subject in $(cat $subject_list_batch_2); do
    echo "Averaging across ERFs for subject $subject"

    # Run the script to average across epochs
    python3 MEG_preprocessing/combine_time_series_from_MEG.py --sub $subject \
    --bids_root $bids_root/Cogitate_Batch2/MEG_Data --region_option hypothesis_driven
done


# Done :)
print("All finished with preprocessing. Ready for time-series feature extraction.")
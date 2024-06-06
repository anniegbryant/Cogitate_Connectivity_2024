#!/usr/bin/env bash

##################################################################################################
# Preprocessing [Artemis, batch array]
##################################################################################################

# Define the batch job array command
input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list.txt

# # Step 1
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_preproc_^array_index^.out \
#    -N All_MEG_preproc \
#    -J 1-48 \
#    -l select=1:ncpus=1:mem=40GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,step=1 \
#    1_preprocess_MEG_subjects.pbs"
# $cmd

# # Step 2
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_preproc_^array_index^.out \
#    -N All_MEG_preproc \
#    -J 1-48 \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,step=2 \
#    1_preprocess_MEG_subjects.pbs"
# $cmd

##################################################################################################
# recon-all [Artemis, individual jobs]
##################################################################################################

# # Define the recon-all command loop
# cat /project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt | while read line 
# do
#    subject=$line
#    cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/recon_all_${subject}.out \
#    -N ${subject}_recon_all \
#    -v subject=$subject \
#    2_recon_all.pbs"

#    # Run the command
#    $cmd
# done

##################################################################################################
# Scalp reconstruction [Physics cluster]
##################################################################################################

# Physics cluster
# input_model_file=/headnode1/abry4213/data/Cogitate_MEG_challenge/subject_list.txt
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/MEG_scalp_recon_^array_index^.out \
#    -N All_scalp_recon \
#    -J 1-48 \
#    -v input_model_file=$input_model_file \
#    3_scalp_recon.pbs"

# $cmd

##################################################################################################
# BEM
##################################################################################################

# input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt

# # Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_BEM_^array_index^.out \
# -N BEM \
# -J 1-46 \
# -v input_model_file=$input_model_file \
# 4_BEM.pbs"

# # Run the command
# $cmd

##################################################################################################
# Subject-specific source localization 
##################################################################################################

# input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt

# # Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_subject_source_localization_^array_index^.out \
# -N subject_source_localization \
# -J 1-46 \
# -v input_model_file=$input_model_file \
# 5_subject_source_localization.pbs"

# # Run the command
# $cmd

##################################################################################################
# Global source localization
##################################################################################################

# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/global_source_localization.out \
# -N global_source_localization \
# 6_global_source_localization.pbs"

# # Run the command
# $cmd

##################################################################################################
# Extract time series and frequency power across participants
##################################################################################################

input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt

# # Define the command
n_jobs=4
num_cores=10
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_extract_time_series_^array_index^.out \
# -N MEG_extract_time_series \
# -J 1-46 \
# -v input_model_file=$input_model_file,n_jobs=$n_jobs \
# -l select=1:ncpus=$num_cores:mem=120GB:mpiprocs=$num_cores \
# 7_extract_time_series.pbs"

# echo $cmd

# # Run the command
# $cmd

##################################################################################################
# Combine time series for participant
##################################################################################################

# input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt

# Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_combine_time_series_^array_index^.out \
# -N MEG_combine_time_series \
# -J 1-46 \
# -v input_model_file=$input_model_file \
# -l select=1:ncpus=1:mem=10GB:mpiprocs=1 \
# 8_combine_time_series.pbs"

# echo $cmd

# # Run the command
# $cmd

# Combine all epoch-averaged results into one zipped file
bids_root=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/
time_series_file_path=$bids_root/derivatives/MEG_time_series

# File compression
cd ${time_series_file_path}
zip ${time_series_file_path}/all_epoch_averaged_time_series.zip sub-*_ses-*_meg_*_all_time_series.csv
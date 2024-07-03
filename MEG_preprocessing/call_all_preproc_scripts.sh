#!/usr/bin/env bash

# User-define batch number
batch_number=$1

# Define the batch job array command
input_model_file=/project/hctsa/annie/github/Cogitate_Connectivity_2024/subject_list_batch${batch_number}.txt
# input_model_file=/headnode1/abry4213/github/Cogitate_Connectivity_2024/subject_list_batch${batch_number}.txt

##################################################################################################
# recon-all [Artemis, individual jobs]
##################################################################################################

# # Define the recon-all command loop
# cat $input_model_file | while read line 
# do
#    subject=$line
#    cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/recon_all_${subject}.out \
#    -N ${subject}_recon_all \
#    -v subject=$subject,batch_number=$batch_number \
#    2_recon_all.pbs"

#    # Run the command
#    $cmd
# done

##################################################################################################
# Preprocessing [Artemis, batch array]
##################################################################################################

# # Step 1
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_MEG_preproc_step1_^array_index^.out \
#    -N Batch${batch_number}_MEG_preproc_1 \
#    -J 1-52 \
#    -l select=1:ncpus=1:mem=40GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,batch_number=$batch_number,step=1 \
#    1_preprocess_MEG_subjects.pbs"
# $cmd

# # Step 2
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_MEG_preproc_step2_^array_index^.out \
#    -N Batch${batch_number}_MEG_preproc_2 \
#    -J 1-52 \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,batch_number=$batch_number,step=2 \
#    1_preprocess_MEG_subjects.pbs"
# $cmd


##################################################################################################
# Scalp reconstruction [Physics cluster]
##################################################################################################

# Physics cluster
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_MEG_scalp_recon_^array_index^.out \
#    -N All_scalp_recon \
#    -J 1-52 \
#    -v input_model_file=$input_model_file,batch_number=$batch_number \
#    3_scalp_recon.pbs"
# $cmd

##################################################################################################
# BEM
##################################################################################################

# # Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_MEG_BEM_^array_index^.out \
# -N BEM \
# -J 1-52 \
# -v input_model_file=$input_model_file,batch_number=$batch_number \
# 4_BEM.pbs"

# # Run the command
# $cmd

##################################################################################################
# Subject-specific source localization 
##################################################################################################

# # Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_subject_source_localization_^array_index^.out \
# -N subject_source_localization \
# -J 1-52 \
# -v input_model_file=$input_model_file,batch_number=$batch_number \
# 5_subject_source_localization.pbs"

# # Run the command
# $cmd

##################################################################################################
# Global source localization
##################################################################################################

# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/global_source_localization.out \
# -N global_source_localization \
# -v batch_number=$batch_number \
# 6_global_source_localization.pbs"

# # Run the command
# $cmd

##################################################################################################
# Extract time series and frequency power across participants
##################################################################################################

# # Define the command
num_cores=10
n_jobs=4
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_extract_time_series_^array_index^.out \
# -N MEG_extract_time_series \
# -J 1-52 \
# -v input_model_file=$input_model_file,batch_number=$batch_number \
# -l select=1:ncpus=$num_cores:mem=120GB:mpiprocs=$num_cores \
# 7_extract_time_series.pbs"

# echo $cmd

# # Run the command
# $cmd

for line_to_read in 14 20; do 
    cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_MEG_extract_time_series_${line_to_read}.out \
    -N Batch${batch_number}_MEG_extract_time_series \
    -l select=1:ncpus=1:mem=80GB:mpiprocs=1 \
    -v line_to_read=$line_to_read,input_model_file=$input_model_file,batch_number=$batch_number,n_jobs=$n_jobs \
    7_extract_time_series.pbs"
    $cmd
done

##################################################################################################
# Combine time series for participant
##################################################################################################

# Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_combine_time_series_^array_index^.out \
# -N MEG_combine_time_series \
# -J 1-52 \
# -v input_model_file=$input_model_file,batch_number=$batch_number \
# -l select=1:ncpus=1:mem=10GB:mpiprocs=1 \
# 8_combine_time_series.pbs"

# echo $cmd

# # Run the command
# $cmd

# for line_to_read in 29; do 
#     cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_MEG_combine_time_series_${line_to_read}.out \
#     -N Batch${batch_number}_MEG_combine_time_series \
#     -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#     -v line_to_read=$line_to_read,input_model_file=$input_model_file,batch_number=$batch_number \
#     8_combine_time_series.pbs"
#     $cmd
# done

# Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_combine_time_series_^array_index^.out \
# -N MEG_combine_time_series \
# -v line_to_read=52,input_model_file=$input_model_file,batch_number=$batch_number \
# -l select=1:ncpus=1:mem=10GB:mpiprocs=1 \
# 8_combine_time_series.pbs"

# echo $cmd

# # Run the command
# $cmd

# # Combine all epoch-averaged results into one zipped file
# bids_root=/project/hctsa/annie/data/Cogitate_Batch${batch_number}/MEG_Data/
# time_series_file_path=$bids_root/derivatives/MEG_time_series

# # File compression
# cd ${time_series_file_path}
# zip ${time_series_file_path}/all_epoch_averaged_time_series.zip sub-*_ses-*_meg_*_all_time_series.csv
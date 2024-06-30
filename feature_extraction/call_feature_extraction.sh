#!/usr/bin/env bash

# User-define batch number
batch_number=$1

# Define the batch job array command
# input_model_file=/project/hctsa/annie/github/Cogitate_Connectivity_2024/subject_list_batch${batch_number}.txt
input_model_file=/headnode1/abry4213/github/Cogitate_Connectivity_2024/subject_list_batch${batch_number}.txt

##################################################################################################
# Running pyspi across subjects, averaged epochs
##################################################################################################

# # Averaged epochs
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_pyspi_averaged_^array_index^_fast.out \
#    -J 1-52 \
#    -N fast_pyspi_MEG \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,batch_number=$batch_number \
#    run_pyspi_for_subject_averaged_epochs.pbs"
# $cmd

##################################################################################################
# Running pyspi across subjects, individual epochs
##################################################################################################

# Individual epochs
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/pyspi_for_individual_epochs_^array_index^_fast.out \
#    -N pyspi_individual_epochs \
#    -J 1-52 \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,batch_number=$batch_number \
#    run_pyspi_for_subject_individual_epochs.pbs"
# $cmd
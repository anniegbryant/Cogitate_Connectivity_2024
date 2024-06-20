#!/usr/bin/env bash

input_model_file=/headnode1/abry4213/data/Cogitate_MEG_challenge/subject_list_filtered.txt

##################################################################################################
# Running pyspi across subjects, averaged epochs
##################################################################################################


# # Averaged epochs
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/feature_extraction_^array_index^_fast.out \
#    -J 1-46 \
#    -N fast_pyspi_MEG \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v n_jobs=1,input_model_file=$input_model_file \
#    run_pyspi_for_subject_averaged_epochs.pbs"
# $cmd

##################################################################################################
# Running pyspi across subjects, averaged epochs
##################################################################################################

# Individual epochs
cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/pyspi_for_individual_epochs_^array_index^_fast.out \
   -N pyspi_individual_epochs \
   -J 1-3 \
   -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
   -v n_jobs=1,input_model_file=$input_model_file \
   run_pyspi_for_subject_individual_epochs.pbs"
$cmd


#!/usr/bin/env bash

##################################################################################################
# Preprocessing [Artemis, batch array]
##################################################################################################

input_model_file=/headnode1/abry4213/data/Cogitate_MEG_challenge/subject_list_filtered.txt

# Step 1
cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/feature_extraction_^array_index^_fast.out \
   -J 1-46 \
   -N fast_pyspi_MEG \
   -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
   -v n_jobs=1,input_model_file=$input_model_file \
   run_pyspi_for_subject_averaged_epochs.pbs"
$cmd

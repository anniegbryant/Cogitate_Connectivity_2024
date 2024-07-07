#!/usr/bin/env bash

# User-define batch number
batch_number=$1

# Define the batch job array command
input_model_file=/project/hctsa/annie/github/Cogitate_Connectivity_2024/subject_list_batch${batch_number}.txt

###################### Averaged epoch classification ##################
n_jobs=10
cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_group_averaged_classification.out \
   -N Batch${batch_number}_averaged_classification \
   -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
   -v input_model_file=$input_model_file,batch_number=$batch_number,n_jobs=$n_jobs \
   run_averaged_classifiers.pbs"
$cmd

###################### Indivivdual epoch classification ##################
# n_jobs=10
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_Batch${batch_number}_individual_epoch_classification_^array_index^.out \
#    -N Batch${batch_number}_individual_classification \
#    -J 1-52 \
#    -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
#    -v input_model_file=$input_model_file,batch_number=$batch_number,n_jobs=$n_jobs \
#    run_individual_classifiers.pbs"
# $cmd
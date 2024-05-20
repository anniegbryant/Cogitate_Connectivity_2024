#!/usr/bin/env bash

##################################################################################################
# Preprocessing 
##################################################################################################

# # Define the batch job array command
# cat /project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt | while read line 
# do
#    subject=$line
#    cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_preproc_${subject}.out \
#    -N ${subject}_MEG_preproc \
#    -v subject=$subject \
#    1_preprocess_MEG_subjects.pbs"

#    # Run the command
#    $cmd
# done

# Physics cluster
# input_model_file=/headnode1/abry4213/data/Cogitate_MEG_challenge/subject_list.txt
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/MEG_preproc_^array_index^.out \
#    -N All_MEG_preproc \
#    -J 1-48 \
#    -v input_model_file=$input_model_file \
#    1_preprocess_MEG_subjects.pbs"

# $cmd

##################################################################################################
# Subject-specific source localization 
##################################################################################################

# input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt

# # Define the command
# cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/MEG_preproc_^array_index^.out \
# -N subject_source_localization \
# -J 1-46 \
# -v input_model_file=$input_model_file \
# 2_subject_source_localization.pbs"

# # Run the command
# $cmd

##################################################################################################
# Global source localization
##################################################################################################

cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/global_source_localization.out \
-N global_source_localization \
3_global_source_localization.pbs"

# # Run the command
# $cmd

##################################################################################################
# Subject-specific activation analysis
##################################################################################################

input_model_file=/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/subject_list_filtered.txt

# Define the command
cmd="qsub -o /project/hctsa/annie/github/Cogitate_Connectivity_2024/cluster_output/subject_activation_analysis_^array_index^.out \
-N subject_activation_analysis \
-J 1-46 \
-v input_model_file=$input_model_file \
4_finish_activation_analysis.pbs"

# Run the command
$cmd
#!/usr/bin/env bash

# Update to where your bids_root directory is
export bids_root=/project/hctsa/annie/data/

######################################################################################
# Fit logistic regression classifiers for (1) stimulus type and (2) task relevance
######################################################################################

# Use 1 job by default, you can increase as your system allows
n_jobs=1

# Batch 1
python3 classification/fit_all_classifiers.py --bids_root ${bids_root}/Cogitate_Batch1/MEG_Data --n_jobs $n_jobs

# Batch 2
python3 classification/fit_all_classifiers.py --bids_root ${bids_root}/Cogitate_Batch2/MEG_Data --n_jobs $n_jobs
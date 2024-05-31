import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import json
import pandas as pd
import pickle
from joblib import Parallel, delayed

import mne
import mne_bids
from mne.minimum_norm import apply_inverse,apply_inverse_epochs

parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='CB040',
                    help='site_id + subject_id (e.g. "CB040")')
parser.add_argument('--bids_root',
                    type=str,
                    default='/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/',
                    help='Path to the BIDS root directory')
parser.add_argument('--region_option',
                    type=str,
                    default='all',
                    help='Set of regions to use ("all" or "hypothesis_driven")')
parser.add_argument('--n_jobs',
                    type=int,
                    default='all',
                    help='Number of concurrent processing jobs')
opt=parser.parse_args()


# Set params
visit_id = "1" # Using the first visit for this project
sfreq = 100 # Setting sampling frequency to 100Hz

subject_id = opt.sub
region_option = opt.region_option
bids_root = opt.bids_root
n_jobs = opt.n_jobs

debug = False

factor = ['Category', 'Task_relevance', "Duration"]
# conditions = [['face', 'object', 'letter', 'false'],
#               ['Relevant target', 'Relevant non-target', 'Irrelevant'],
#               ['500ms', '1000ms', '1500ms']]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant target', 'Relevant non-target', 'Irrelevant'],
              ['1000ms']]

# Helper function to create a dictionary of ROI labels depending on the type of region subset requested
def compute_ROI_labels(labels_atlas, region_option, rois_deriv_root):
    # Create dictionary to store labels and vertices
    labels_dict = {}
    if region_option == 'hypothesis_driven':
        # Read GNW and IIT ROI list
        f = open(op.join(rois_deriv_root,
                        'hypothesis_driven_ROIs.json'))
        hypothesis_driven_ROIs = json.load(f)

        # GNWT ROIs
        print("GNWT ROIs:")
        for lab in hypothesis_driven_ROIs['GNWT_ROIs']:
            print(lab)
            labels_dict["GNWT_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])

        # IIT ROIs
        print("IIT ROIs")
        for lab in hypothesis_driven_ROIs['IIT_ROIs']:
            print(lab)
            labels_dict["IIT_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])

        # Category-selective ROIs
        print("Category-selective ROIs:")
        for lab in hypothesis_driven_ROIs['Category_Selective_ROIs']:
            print(lab)
            labels_dict["Category_Selective_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])

        # Merge all labels in a single one separatelly for GNW and IIT 
        labels_dict['GNWT_meta_ROI'] = np.sum([l for l_name, l in labels_dict.items() if 'GNWT' in l_name])
        labels_dict['IIT_meta_ROI'] = np.sum([l for l_name, l in labels_dict.items() if 'IIT' in l_name])
        labels_dict['Category_Selective_meta_ROI'] = np.sum([l for l_name, l in labels_dict.items() if 'Category_Selective' in l_name])

        # Only keep the meta-ROIs
        labels_dict = {k: v for k, v in labels_dict.items() if 'meta_ROI' in k}

    else:
        for label in labels_atlas: 
            label_name = label.name
            labels_dict[label_name] =  np.sum([label])

    return labels_dict

# Helper function to compute covariance matrices and inverse solution 
def fit_cov_and_inverse(subject_id, visit_id, factor, conditions, bids_root, downsample=True, tmin=-0.5, tmax=1.99):
    # Set directory paths
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    source_deriv_root = op.join(bids_root, "derivatives", "source_dur_ERF")

    if not op.exists(source_deriv_root):
        os.makedirs(source_deriv_root, exist_ok=True)

    source_figure_root =  op.join(source_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(source_figure_root):
        os.makedirs(source_figure_root)

    # Set task
    bids_task = 'dur'
    
    # Read epoched data
    bids_path_epo = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo',
        extension='.fif',
        check=False)
    
    bids_path_epo_rs = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo_rs',
        extension='.fif',
        check=False)
    
    print("Loading epochs data")
    epochs_all = mne.read_epochs(bids_path_epo.fpath, preload=True)

    if not downsample:
        epochs_final = epochs_all

    # If downsampling is requested
    else:
        print("Applying downsampling")
        if os.path.exists(bids_path_epo_rs.fpath):
            epochs_rs = mne.read_epochs(bids_path_epo_rs.fpath,
                                    preload=True)
        else:
            epochs_all = mne.read_epochs(bids_path_epo.fpath,
                                    preload=True)
            resample_epochs(epochs_all, sfreq, bids_path_epo_rs, tmin=tmin, tmax=tmax)
            epochs_rs = mne.read_epochs(bids_path_epo_rs.fpath, preload=True)
        epochs_final = epochs_rs

    # Run baseline correction
    print("Running baseline correction")
    b_tmin = tmin
    b_tmax = 0.
    baseline = (b_tmin, b_tmax)
    epochs_final.apply_baseline(baseline=baseline)

    # Compute rank
    print("Computing the rank")
    if os.path.isfile(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_rank.pkl"):
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_rank.pkl", 'rb') as f:
            rank = pickle.load(f)
    else: 
        rank = mne.compute_rank(epochs_final, 
                                tol=1e-6, 
                                tol_kind='relative')
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_rank.pkl", 'wb') as f:
            pickle.dump(rank, f)

    # Read forward model
    print("Reading forward model")
    bids_path_fwd = bids_path_epo.copy().update(
            root=fwd_deriv_root,
            task=bids_task,
            suffix="surface_fwd",
            extension='.fif',
            check=False)
    fwd = mne.read_forward_solution(bids_path_fwd.fpath)

    # Compute covariance matrices
    print("Computing covariance matrices")
    if os.path.isfile(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_common_cov.pkl"): 
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_common_cov.pkl", 'rb') as f:
            common_cov = pickle.load(f)
    else:
        base_cov = mne.compute_covariance(epochs_final, 
                                        tmin=-0.5, 
                                        tmax=0, 
                                        n_jobs=n_jobs,
                                        method='empirical', 
                                        rank=rank)

        active_cov = mne.compute_covariance(epochs_final, 
                                        tmin=0,
                                        tmax=None,
                                        n_jobs=n_jobs,
                                        method='empirical', 
                                        rank=rank)
        common_cov = base_cov + active_cov

        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_common_cov.pkl", 'wb') as f:
            pickle.dump(common_cov, f)

    # Make inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs_final.info,
        fwd, 
        common_cov,
        loose=.2,
        depth=.8,
        fixed=False,
        rank=rank,
        use_cps=True)

    # Find all combinations between variables' levels
    if len(factor) == 1:
        cond_combs = list(itertools.product(conditions[0]))
    if len(factor) == 2:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1]))
    if len(factor) == 3:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1],
                                            conditions[2]))
        
    return epochs_final, inverse_operator, cond_combs
# Helper function to process condition combination 
def cond_comb_helper_process_by_epoch(cond_comb, epochs_final, inverse_operator, labels_dict, subject_time_series_output_path):
    print("\nAnalyzing %s: %s" % (factor, cond_comb))

    # Take subset of epochs corresponding to this condition combination
    cond_epochs = epochs_final['%s == "%s" and %s == "%s" and %s == "%s"' % (
        factor[0], cond_comb[0], 
        factor[1], cond_comb[1], 
        factor[2], cond_comb[2])]
    fname_base = f"{cond_comb[0]}_{cond_comb[1]}_{cond_comb[2]}".replace(" ","-")

    # Compute inverse solution for each epoch
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(cond_epochs, inverse_operator,
                                lambda2=1.0 / snr ** 2, verbose=False,
                                method="dSPM", pick_ori="normal")

    # Extract time course for each stc
    for i in range(len(stcs)):

        # Find epoch number
        epoch_number = i+1

        # Find stc
        stc = stcs[i]

        # Loop over labels        
        for label_name, label in labels_dict.items():

            # Select data in label
            stc_in = stc.in_label(label)

            # Extract time course data, averaged across channels within ROI
            times = stc_in.times
            data = stc_in.data.mean(axis=0)

            # Concatenate into dataframe
            epoch_df = pd.DataFrame({
                'epoch_number': epoch_number,
                'stimulus_type': cond_comb[0], 
                'relevance_type': cond_comb[1],
                'duration': cond_comb[2],
                'times': times,
                'meta_ROI': label_name,
                'data': data})
            
            # Write this epoch to a CSV file
            output_CSV_file = op.join(subject_time_series_output_path, f"{fname_base}_epoch{epoch_number}_{label_name}.csv")
            epoch_df.to_csv(output_CSV_file, index=False)
                
# Extract all epoch time series
def extract_all_epoch_TS(subject_id, visit_id, region_option, factor, conditions):

    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")

    time_series_output_path = op.join(bids_root, "derivatives", "MEG_time_series")
    if not op.exists(time_series_output_path):
        os.makedirs(time_series_output_path, exist_ok=True)

    # Time series output path for this subject
    subject_time_series_output_path = op.join(time_series_output_path, f"sub-{subject_id}", f"ses-{visit_id}", "meg")
    if not op.exists(subject_time_series_output_path):
        os.makedirs(subject_time_series_output_path, exist_ok=True)
        
    # Set task
    bids_task = 'dur'

    # Read epoched data
    bids_path_epo = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo',
        extension='.fif',
        check=False)
 
    # Use Desikan--Killiany atlas to compute dictionary of labels
    labels_atlas = mne.read_labels_from_annot(
        "sub-"+subject_id, 
        parc='aparc.a2009s',
        subjects_dir=fs_deriv_root)
    labels_dict = compute_ROI_labels(labels_atlas, region_option, rois_deriv_root)

    # Save label names
    bids_path_label_names = bids_path_epo.copy().update(
                    root=time_series_output_path,
                    suffix="desc-labels_"+region_option,
                    extension='.txt',
                    check=False)

    # Find epochs_rs, inverse_operator, cond_combs
    print("Now finding inverse operator")
    epochs_final, inverse_operator, cond_combs = fit_cov_and_inverse(subject_id, visit_id, factor, conditions, bids_root, downsample=False)

    # extract time course in label with pca_flip mode
    src = inverse_operator['src']

    # Loop over conditions of interest
    print("Now looping over task conditions")
    Parallel(n_jobs=int(n_jobs))(delayed(cond_comb_helper_process_by_epoch)(cond_comb=cond_comb, 
                                                                epochs_final=epochs_final, 
                                                                inverse_operator=inverse_operator, 
                                                                labels_dict=labels_dict, 
                                                                subject_time_series_output_path=subject_time_series_output_path)
                                                for cond_comb in cond_combs)
    
    if not os.path.isfile(bids_path_label_names):
        with open(bids_path_label_names.fpath, "w") as output:
            output.write(str(list(labels_dict.keys())))

if __name__ == '__main__':
    extract_all_epoch_TS(subject_id, visit_id, region_option, factor, conditions)
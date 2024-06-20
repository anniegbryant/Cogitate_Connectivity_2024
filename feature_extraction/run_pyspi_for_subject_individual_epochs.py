import pandas as pd
import numpy as np
import pyspi
from pyspi.calculator import Calculator
import os.path as op
import os
from copy import deepcopy
from joblib import Parallel, delayed
import argparse
import glob

parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='CB040',
                    help='site_id + subject_id (e.g. "CB040")')
parser.add_argument('--visit_id',
                    type=str,
                    default='1',
                    help='Visit ID (e.g., 1)')
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
                    default=1,
                    help='Number of concurrent processing jobs')
opt=parser.parse_args()

subject_id = opt.sub 
bids_root = opt.bids_root
region_option = opt.region_option
visit_id = opt.visit_id
n_jobs = opt.n_jobs
duration="1000ms"

# Time series output path for this subject
time_series_path = op.join(bids_root, "derivatives", "MEG_time_series")
output_feature_path = op.join(bids_root, "derivatives", "time_series_features")

# Define ROI lookup table
if region_option == "hypothesis_driven":
    ROI_lookup = {"proc-0": "Category_Selective",
              "proc-1": "GNWT",
              "proc-2": "IIT"}
    
if op.isfile(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_pyspi_results_individual_epochs_{duration}.csv"):
    print(f"SPI results for sub-{subject_id} already exist. Skipping.")
    exit() 

# Initialise a base calculator
calc = Calculator(subset='fast')

all_epoch_files = [f for f in os.listdir(f"{time_series_path}/sub-{subject_id}_ses-{visit_id}_epochs") if "GNWT" in f]
meta_ROI_list = ["IIT", "GNWT", "Category_Selective"]

all_pyspi_results_for_this_subject_list = []

for csv_file in all_epoch_files:
    stimulus_type, relevance, duration, epoch_number = csv_file.split("_")[0:4]

    results_across_ROIs_list = []

    # Combine results across IIT, GNWT, and Category-Selective meta-ROIs
    for meta_ROI in meta_ROI_list:
        this_ROI_data = pd.read_csv(f"{time_series_path}/sub-{subject_id}_ses-{visit_id}_epochs/{stimulus_type}_{relevance}_{duration}_{epoch_number}_{meta_ROI}_meta_ROI.csv")
        this_ROI_data['duration'] = this_ROI_data['duration'].str.replace('ms', '').astype(int)
        this_ROI_data['times'] = np.round(this_ROI_data['times']*1000)
        this_ROI_data['times'] = this_ROI_data['times'].astype(int)

        # Filter times >= 0
        this_ROI_data = this_ROI_data.query('times >= 0')

        # Assign stimulus as on if times < duration and off if times >= duration
        this_ROI_data['stimulus'] = np.where(this_ROI_data['times'] < this_ROI_data['duration'], 'on', 'off')

        # Append to list
        results_across_ROIs_list.append(this_ROI_data)

    results_across_ROIs = pd.concat(results_across_ROIs_list, axis=0).reset_index()

    # Iterate over onset vs. offset
    for stimulus_presentation in ['on', 'off']:
        results_across_ROIs_with_this_stim = results_across_ROIs.query("stimulus==@stimulus_presentation")

        # Pivot so that the columns are meta_ROI and the rows are data
        df_wide = (results_across_ROIs_with_this_stim.filter(items=['times', 'meta_ROI', 'data'])
                        .reset_index()
                        .pivot(index='meta_ROI', columns='times', values='data'))

        # Make deepcopy of calc 
        calc_copy = deepcopy(calc)

        # Convert to numpy array
        TS_array = df_wide.to_numpy()

        # Load data 
        calc_copy.load_dataset(TS_array)
        calc_copy.compute()

        SPI_res = deepcopy(calc_copy.table)

        # Iterate over each SPI
        SPI_res.columns = SPI_res.columns.to_flat_index()

        SPI_res = SPI_res.rename(columns='__'.join).assign(meta_ROI_from = lambda x: x.index)
        SPI_res_long = SPI_res.melt(id_vars='meta_ROI_from', var_name='SPI__meta_ROI_to', value_name='value')

        SPI_res_long["SPI"] = SPI_res_long["SPI__meta_ROI_to"].str.split("__").str[0]
        SPI_res_long["meta_ROI_to"] = SPI_res_long["SPI__meta_ROI_to"].str.split("__").str[1]

        SPI_res_long = (SPI_res_long
                        .drop(columns='SPI__meta_ROI_to')
                        .query('meta_ROI_from != meta_ROI_to')
                        .assign(meta_ROI_from = lambda x: x['meta_ROI_from'].map(ROI_lookup),
                                meta_ROI_to = lambda x: x['meta_ROI_to'].map(ROI_lookup))
                        .filter(items=['SPI', 'meta_ROI_from', 'meta_ROI_to', 'value'])
                        .assign(stimulus_type = stimulus_type,
                                stimulus_presentation = stimulus_presentation,
                                relevance_type = relevance,
                                duration = duration,
                                epoch_number = epoch_number,
                                subject_ID = subject_id)
        )

        all_pyspi_results_for_this_subject_list.append(SPI_res_long)

all_pyspi_results_for_this_subject = pd.concat(all_pyspi_results_for_this_subject_list, axis=0).reset_index()
all_pyspi_results_for_this_subject.to_csv(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_pyspi_results_individual_epochs_{duration}.csv", index=False)
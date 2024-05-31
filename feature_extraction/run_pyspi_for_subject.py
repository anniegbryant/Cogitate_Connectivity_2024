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
    
if op.isfile(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_pyspi_results_{duration}.csv"):
    print(f"SPI results for sub-{subject_id} already exist. Skipping.")
    exit() 

# Iterate over all the time-series files for this subject
sample_TS_data_list = []

sample_TS_data=pd.read_csv(f"{time_series_path}/sub-{subject_id}_ses-{visit_id}_meg_{duration}_all_time_series.csv")
sample_TS_data['duration'] = sample_TS_data['duration'].str.replace('ms', '').astype(int)
sample_TS_data['times'] = np.round(sample_TS_data['times']*1000)
sample_TS_data['times'] = sample_TS_data['times'].astype(int)

# Filter times to the duration range
sample_TS_data_onset = sample_TS_data.query('times >= 0 and times < @sample_TS_data.duration')
sample_TS_data_offset = sample_TS_data.query('times >= @sample_TS_data.duration')

# Create list of dataframes for each stimulus_type, relevance_type, duration, and frequency_band
# One list for 'on' (while stimulus is being presented) and another for 'off' (after stimulus is no longer being presented)
on_sample_TS_data_list = []
off_sample_TS_data_list = []
for stimulus_type in sample_TS_data_onset['stimulus_type'].unique():
    for relevance_type in sample_TS_data_onset['relevance_type'].unique():
        for duration in [1000]:
        # for duration in sample_TS_data_onset['duration'].unique():
            TS_data_on = sample_TS_data_onset.query('stimulus_type == @stimulus_type and relevance_type == @relevance_type and duration == @duration')
            if TS_data_on.empty:
                print(f"Missing data for {stimulus_type}, {relevance_type}, {duration}")
            on_sample_TS_data_list.append(TS_data_on)
            TS_data_off = sample_TS_data_offset.query('stimulus_type == @stimulus_type and relevance_type == @relevance_type and duration == @duration')
            off_sample_TS_data_list.append(TS_data_off)

def run_pyspi_for_df(subject_id, df, calc):
        # Make deepcopy of calc 
        calc_copy = deepcopy(calc)

        # Pivot so that the columns are meta_ROI and the rows are data
        df_wide = (df.filter(items=['times', 'Category_Selective', 'GNWT', 'IIT'])
                     .melt(id_vars='times', var_name='meta_ROI', value_name='data')
                     .reset_index()
                     .pivot(index='meta_ROI', columns='times', values='data'))

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
                        .assign(stimulus_type = df['stimulus_type'].unique()[0],
                                relevance_type = df['relevance_type'].unique()[0],
                                duration = df['duration'].unique()[0],
                                subject_ID = subject_id)
        )

        return SPI_res_long
# Initialise an empty list for the results
on_data_pyspi_list = []
off_data_pyspi_list = []

# Initialise a base calculator
calc = Calculator(subset='fast')

# # Run for "on" data
# on_data_pyspi_list = Parallel(n_jobs=n_jobs)(delayed(run_pyspi_for_df)(df=on_data, 
#                                                             calc=calc)
#                                             for on_data in on_sample_TS_data_list)

# on_data_pyspi_res = pd.concat(on_data_pyspi_list)

# # Run for "off" data
# off_data_pyspi_list = Parallel(n_jobs=n_jobs)(delayed(run_pyspi_for_df)(df=off_data, 
#                                                             calc=calc)
#                                             for off_data in off_sample_TS_data_list)

# off_data_pyspi_res = pd.concat(off_data_pyspi_list)

# Run for "on" data
for on_data in on_sample_TS_data_list:
    on_data_pyspi = run_pyspi_for_df(subject_id, on_data, calc).assign(stimulus = "on")
    on_data_pyspi_list.append(on_data_pyspi)
on_data_pyspi_res = pd.concat(on_data_pyspi_list)

# Run for "off" data
for off_data in off_sample_TS_data_list:
    off_data_pyspi = run_pyspi_for_df(subject_id, off_data, calc).assign(stimulus = "off")
    off_data_pyspi_list.append(off_data_pyspi)
off_data_pyspi_res = pd.concat(off_data_pyspi_list)

# Concatenate the results and save to a feather file
all_pyspi_res = pd.concat([on_data_pyspi_res, off_data_pyspi_res]).reset_index() 
all_pyspi_res.to_csv(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_pyspi_results_1000ms.csv", index=False)
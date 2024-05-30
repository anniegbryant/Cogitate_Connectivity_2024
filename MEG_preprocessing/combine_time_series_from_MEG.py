import os
import os.path as op
import numpy as np
import argparse
import pandas as pd

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
opt=parser.parse_args()


# Set params
visit_id = "1" # Using the first visit for this project
duration  ="1000ms"

subject_id = opt.sub
region_option = opt.region_option
bids_root = opt.bids_root

# Function to zip the desired files for the participant
def combine_files_for_participant(subject_id, visit_id, bids_root, duration="1000ms"):

    output_file = f"{bids_root}/derivatives/MEG_time_series/sub-{subject_id}_ses-{visit_id}_meg_{duration}_all_time_series.csv"

    if not os.path.isfile(output_file):
        time_series_output_path = f"{bids_root}/derivatives/MEG_time_series"
        subject_time_series_output_path = f"{time_series_output_path}/sub-{subject_id}/ses-{visit_id}/meg"

        time_series_files = [f"{subject_time_series_output_path}/{file}" for file in os.listdir(subject_time_series_output_path) if f"{duration}_epoch" in file]
        all_time_series_data = pd.concat([pd.read_csv(file) for file in time_series_files])

        # Average across epochs
        averaged_by_condition = (all_time_series_data
                                .groupby(["stimulus_type", "relevance_type", "duration", "times", "meta_ROI"], as_index=False)["data"]
                                .mean()
                                .assign(meta_ROI = lambda x: x.meta_ROI.str.replace("_meta_ROI", ""))
                                .pivot(index=["stimulus_type", "relevance_type", "duration", "times"], columns="meta_ROI", values="data"))

        averaged_by_condition.columns = averaged_by_condition.columns.get_level_values(0)
        averaged_by_condition.reset_index(inplace=True)

        # Save to CSV
        averaged_by_condition.to_csv(output_file, index=False)


if __name__ == '__main__':
    combine_files_for_participant(subject_id, visit_id, bids_root, duration=duration)
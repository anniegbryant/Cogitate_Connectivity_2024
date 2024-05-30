import os
import os.path as op
import numpy as np
import shutil
import argparse
import zipfile
import glob

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
sfreq = 100 # Setting sampling frequency to 100Hz

subject_id = opt.sub
region_option = opt.region_option
bids_root = opt.bids_root

debug = False

factor = ['Category', 'Task_relevance', "Duration"]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant non-target', 'Irrelevant'],
              ['1000ms']]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant non-target', 'Irrelevant'],
              ['500ms', '1000ms', '1500ms']]

# Function to zip the desired files for the participant
def zip_files_for_participant(subject_id, visit_id, bids_root):
    
    time_series_output_path = f"{bids_root}/derivatives/MEG_time_series"
    subject_time_series_output_path = f"{time_series_output_path}/sub-{subject_id}/ses-{visit_id}/meg"

    # Find the 1000ms files for this participant
    time_series_files = [f"{subject_time_series_output_path}/{file}" for file in op.listdir(subject_time_series_output_path) if "1000ms_epoch" in file]

    output_zip_archive = f"{time_series_output_path}/sub-{subject_id}_ses-{visit_id}_meg.zip"
    ZipFile = zipfile.ZipFile(f"zip {output_zip_archive}", "w" )

    # Create the zip archive if it doesn't already exist
    if not op.isfile(output_zip_archive):
        ZipFile.write(time_series_files, compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    zip_files_for_participant(subject_id, visit_id, bids_root)
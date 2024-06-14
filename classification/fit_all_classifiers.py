import os
import numpy as np
import nibabel as nib
import pandas as pd
import sys
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
import numpy as np
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import itertools

# add path to classification analysis functions
from mixed_sigmoid_normalisation import MixedSigmoidScaler

# Define data paths
pyspi_res_path = "/Users/abry4213/data/Cogitate_MEG_challenge/derivatives/time_series_features"
classification_res_path = "/Users/abry4213/data/Cogitate_MEG_challenge/derivatives/classification_results"

SPI_directionality_info = pd.read_csv("/Users/abry4213/github/Cogitate_Connectivity_2024/feature_extraction/pyspi_SPI_info.csv")

# Load in pyspi results
all_pyspi_res_list = []
for pyspi_res_file in os.listdir(pyspi_res_path):
    pyspi_res = pd.read_csv(f"{pyspi_res_path}/{pyspi_res_file}")
    all_pyspi_res_list.append(pyspi_res)
all_pyspi_res = pd.concat(all_pyspi_res_list)

# Define comparisons

# meta-ROI comparisons: GWNT --> CS, CS --> GNWT, IIT --> CS, CS --> IIT
meta_roi_comparisons = [("GNWT", "Category_Selective"), ("Category_Selective", "GNWT"), ("IIT", "Category_Selective"), ("Category_Selective", "IIT")]

# Relevance type comparisons
relevance_type_comparisons = ["Relevant non-target", "Irrelevant"]

# Stimulus presentation comparisons
stimulus_presentation_comparisons = ["on", "off"]

# Stimulus type comparisons
stimulus_types = all_pyspi_res.stimulus_type.unique().tolist()
stimulus_type_comparisons = list(itertools.combinations(stimulus_types, 2))

# Comparing between stimulus types
if not os.path.isfile(f"{classification_res_path}/comparing_between_stimulus_types_classification_results.csv"):
    # All comparisons list
    comparing_between_stimulus_types_classification_results_list = []

    for meta_roi_comparison in meta_roi_comparisons:
        print("ROI Comparison:" + str(meta_roi_comparison))
        ROI_from, ROI_to = meta_roi_comparison
        for relevance_type in relevance_type_comparisons:
            print("Relevance type:" + str(relevance_type))
            for stimulus_presentation in stimulus_presentation_comparisons:
                print("Stimulus presentation:" + str(stimulus_presentation))
                # Finally, we get to the final dataset
                final_dataset_for_classification = all_pyspi_res.query("meta_ROI_from == @ROI_from & meta_ROI_to == @ROI_to & relevance_type == @relevance_type & stimulus == @stimulus_presentation").reset_index(drop=True).drop(columns=['index'])

                for SPI in final_dataset_for_classification.SPI.unique():

                    # Extract this SPI
                    this_SPI_data = final_dataset_for_classification.query(f"SPI == '{SPI}'")

                    # Find overall number of rows
                    num_rows = this_SPI_data.shape[0]

                    # Extract SPI values
                    this_column_data = this_SPI_data["value"]

                    # Find number of NaN in this column 
                    num_NaN = this_column_data.isna().sum()
                    prop_NaN = num_NaN / num_rows

                    # Find mode and SD
                    column_mode_max = this_column_data.value_counts().max()
                    column_SD = this_column_data.std()

                    # If 0% < num_NaN < 10%, impute by the mean of each component
                    if 0 < prop_NaN < 0.1:
                        values_imputed = (this_column_data
                                            .transform(lambda x: x.fillna(x.mean())))

                        this_column_data = values_imputed
                        print(f"Imputing column values for {SPI}")
                        this_SPI_data["value"] = this_column_data

                    # If there are: 
                    # - more than 10% NaN values;
                    # - more than 90% of the values are the same; OR
                    # - the standard deviation is less than 1*10**(-10)
                    # then remove the column
                    if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                        print(f"{SPI} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                        continue
                    
                    # Start an empty list for the classification results
                    SPI_combo_res_list = []
                
                    # Iterate over stimulus combos
                    for this_combo in stimulus_type_comparisons:

                        # Extract just GNWT/CS data first
                        final_dataset_for_classification_this_combo = this_SPI_data.query(f"stimulus_type in {this_combo}'")

                        # Define classification model
                        model = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=127)
                        pipe = Pipeline([('scaler', MixedSigmoidScaler(unit_variance=True)), 
                                                ('model', model)])

                        # Fit classifier
                        X = final_dataset_for_classification_this_combo.value.to_numpy().reshape(-1, 1)
                        y = final_dataset_for_classification_this_combo.stimulus_type.to_numpy().reshape(-1, 1)
                        groups = final_dataset_for_classification_this_combo.subject_ID.to_numpy().reshape(-1, 1)

                        group_stratified_CV = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state=127)

                        this_classifier_res = cross_validate(pipe, X, y, groups=groups, cv=group_stratified_CV, scoring="accuracy", n_jobs=1, 
                                                                    return_estimator=False, return_train_score=False)["test_score"].mean()
                        
                        this_SPI_combo_df = pd.DataFrame({"SPI": [SPI], 
                                                            "meta_ROI_from": [ROI_from],
                                                            "meta_ROI_to": [ROI_to],
                                                            "relevance_type": [relevance_type],
                                                            "stimulus_presentation": [stimulus_presentation],
                                                            "stimulus_combo": [this_combo], 
                                                            "accuracy": [this_classifier_res]})
                        
                        # Append to growing results list
                        comparing_between_stimulus_types_classification_results_list.append(this_SPI_combo_df)

    comparing_between_stimulus_types_classification_results = pd.concat(comparing_between_stimulus_types_classification_results_list).reset_index(drop=True)
    comparing_between_stimulus_types_classification_results.to_csv(f"{classification_res_path}/comparing_between_stimulus_types_classification_results.csv", index=False)


# Comparing between relevance types
if not os.path.isfile(f"{classification_res_path}/comparing_between_relevance_types_classification_results.csv"):
    # All comparisons list
    comparing_between_relevance_types_classification_results_list = []

    for meta_roi_comparison in meta_roi_comparisons:
        print("ROI Comparison:" + str(meta_roi_comparison))
        ROI_from, ROI_to = meta_roi_comparison
        for stimulus_presentation in stimulus_presentation_comparisons:
            print("Stimulus presentation:" + str(stimulus_presentation))
            # Finally, we get to the final dataset
            final_dataset_for_classification = all_pyspi_res.query("meta_ROI_from == @ROI_from & relevance_type in @relevance_type_comparisons and meta_ROI_to == @ROI_to & stimulus == @stimulus_presentation").reset_index(drop=True).drop(columns=['index'])

            for SPI in final_dataset_for_classification.SPI.unique():

                # Extract this SPI
                this_SPI_data = final_dataset_for_classification.query(f"SPI == '{SPI}'")

                # Find overall number of rows
                num_rows = this_SPI_data.shape[0]

                # Extract SPI values
                this_column_data = this_SPI_data["value"]

                # Find number of NaN in this column 
                num_NaN = this_column_data.isna().sum()
                prop_NaN = num_NaN / num_rows

                # Find mode and SD
                column_mode_max = this_column_data.value_counts().max()
                column_SD = this_column_data.std()

                # If 0% < num_NaN < 10%, impute by the mean of each component
                if 0 < prop_NaN < 0.1:
                    values_imputed = (this_column_data
                                        .transform(lambda x: x.fillna(x.mean())))

                    this_column_data = values_imputed
                    print(f"Imputing column values for {SPI}")
                    this_SPI_data["value"] = this_column_data

                # If there are: 
                # - more than 10% NaN values;
                # - more than 90% of the values are the same; OR
                # - the standard deviation is less than 1*10**(-10)
                # then remove the column
                if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                    print(f"{SPI} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                    continue

                # Start an empty list for the classification results
                SPI_combo_res_list = []

                # Define classification model
                model = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=127)
                pipe = Pipeline([('scaler', MixedSigmoidScaler(unit_variance=True)), 
                                        ('model', model)])

                # Fit classifier
                X = this_SPI_data.value.to_numpy().reshape(-1, 1)
                y = this_SPI_data.relevance_type.to_numpy().reshape(-1, 1)
                groups = this_SPI_data.subject_ID.to_numpy().reshape(-1, 1)

                group_stratified_CV = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state=127)

                this_classifier_res = cross_validate(pipe, X, y, groups=groups, cv=group_stratified_CV, scoring="accuracy", n_jobs=1, 
                                                            return_estimator=False, return_train_score=False)["test_score"].mean()
                
                this_SPI_relevance_results_df = pd.DataFrame({"SPI": [SPI], 
                                                    "meta_ROI_from": [ROI_from],
                                                    "meta_ROI_to": [ROI_to],
                                                    "stimulus_presentation": [stimulus_presentation],
                                                    "comparison": ["Relevant non-target vs. Irrelevant"], 
                                                    "accuracy": [this_classifier_res]})
                
                # Append to growing results list
                comparing_between_relevance_types_classification_results_list.append(this_SPI_relevance_results_df)

    comparing_between_relevance_types_classification_results = pd.concat(comparing_between_relevance_types_classification_results_list).reset_index(drop=True)
    comparing_between_relevance_types_classification_results.to_csv(f"{classification_res_path}/comparing_between_relevance_types_classification_results.csv", index=False)


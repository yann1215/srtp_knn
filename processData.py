# function: include data process functions
# variable: data input and output(label) path
# update: 2024/5/14

import pandas as pd
import glob
import os


def process_data(file_target):
    file_name = data_path + "*" + file_target + "*"
    combined_temp = pd.DataFrame()

    for file in glob.glob(file_name):
        # process data
        raw_data = pd.read_csv(file)
        raw_data["id"] = file_target

        # save new data
        file_new = file.replace("input", "label")
        raw_data.to_csv(file_new, index=False)
        print(file)

        # get combined csv
        combined_temp = pd.concat([combined_temp, raw_data], ignore_index=True)

    return combined_temp


# ---------- Data Path ----------
# caution: notice function process_data() when changing data_path
data_path = "D:/srtp/knn/testPy/input/"

# ---------- Output Path ----------
# caution: notice function process_data() when changing data_path_output
data_path_output = "D:/srtp/knn/testPy/label/"
# check if output folder exists
if not os.path.exists(data_path_output):
    os.makedirs(data_path_output)  # create output folder

# create combined csv
combined_csv = pd.DataFrame()

# ---------- Process Data ----------
combined_temp = process_data("EC")
combined_csv = pd.concat([combined_csv, combined_temp], ignore_index=True)

combined_temp = process_data("KP")
combined_csv = pd.concat([combined_csv, combined_temp], ignore_index=True)

combined_temp = process_data("SA")
combined_csv = pd.concat([combined_csv, combined_temp], ignore_index=True)

# ---------- Save Combined Csv ----------
combined_path = data_path.replace("input/", "") + "data.csv"
combined_csv.to_csv(combined_path, index=False)

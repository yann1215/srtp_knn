import csv
import pandas as pd
import glob
import os


def process_data(file_target):
    file_name = data_path + "*" + file_target + "*"
    for file in glob.glob(file_name):
        raw_data = pd.read_csv(file)
        raw_data["id"] = file_target
        fil_new = file.replace("input", "label")
        raw_data.to_csv(fil_new, index=False)
        print(file)


def add_label(data, target):
    data["id"] = target
    return data


# data path
# caution: notice function process_data() when changing data_path
data_path = "D:/srtp/knn/testPy/input/"

# output path
# caution: notice function process_data() when changing data_path_output
data_path_output = "D:/srtp/knn/testPy/label/"
# check if output folder exists
if not os.path.exists(data_path_output):
    os.makedirs(data_path_output)  # create output folder

# keywords
process_data("EC")
process_data("KP")
process_data("SA")

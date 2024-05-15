# update: 2024/5/15

import pandas as pd
import glob
import os


def process_data(file_target):
    file_name = data_path + "*" + file_target + "*"
    temp = pd.DataFrame()

    for file in glob.glob(file_name):
        # process data
        raw_data = pd.read_csv(file)

        if "spotSTD/Mean" not in raw_data.columns:
            raw_data["spotSTD/Mean"] = raw_data["spotSTD"] / raw_data["spotMean"]

        if "id" not in raw_data.columns:
            raw_data["id"] = file_target

        # save new data
        file_new = file.replace("input", "label")
        raw_data.to_csv(file_new, index=False)
        # print(file)

        # get combined csv
        temp = pd.concat([temp, raw_data], ignore_index=True)

    return temp


if __name__ == '__main__':
    # ---------- Data Path ----------
    # caution: notice function process_data() when changing data_path
    data_path = "D:/srtp/knn/data/input/"
    data_path_output = "D:/srtp/knn/data/label/"
    if not os.path.exists(data_path_output):
        os.makedirs(data_path_output)  # create output folder

    id_feature = ["EC", "KP", "SA"]

    # ---------- Process Data ----------
    # create combined csv
    combined_csv = pd.DataFrame()

    for feature in id_feature:
        combined_temp = process_data(feature)
        combined_csv = pd.concat([combined_csv, combined_temp], ignore_index=True)

    # ---------- Save Combined Csv ----------
    combined_path = data_path.replace("input/", "") + "data.csv"
    combined_csv.to_csv(combined_path, index=False)

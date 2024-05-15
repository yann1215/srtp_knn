# update: 2024/5/15

import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import os

# Main
if __name__ == '__main__':

    # ---------- Initialize ----------
    # run data process function
    os.system("python {}".format("processData.py"))
    os.system("python {}".format("processTest.py"))

    # data path
    csv_path = "D:/srtp/knn/data/data.csv"
    test_path = "D:/srtp/knn/data/test.csv"

    # features used in knn model
    # full features include: "spotMean", "spotSTD", "speedMean", "speedSTD", "spotMax", "spotSTD/Mean"
    knn_feature = ["spotMean", "speedMean", "speedSTD", "spotMax", "spotSTD/Mean"]

    # ---------- Load Train Data ----------
    full_csv = pd.read_csv(csv_path, index_col=0)

    full_data = full_csv.loc[:, knn_feature]

    full_label = full_csv.loc[:, "id"]

    data_train = full_data
    label_train = full_label
    # data_train, data_test, label_train, label_test \
    #     = train_test_split(full_data, full_label, test_size=0.3, random_state=None)

    # ---------- Fit Knn Model ----------
    # knn model name: knn
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(data_train, label_train)

    # ---------- Load Test Data ----------
    # full_data include: "spotMean", "spotSTD", "speedMean", "speedSTD", "spotMax"
    full_test = pd.read_csv(test_path, index_col=0)

    data_test = full_test.loc[:, knn_feature]

    label_test = full_test.loc[:, "id"]

    # ---------- Predict Label ----------
    label_predict = knn.predict(data_test)

    # ---------- Calculate Accuracy ----------
    accuracy = accuracy_score(label_test, label_predict)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:")
    print(classification_report(label_test, label_predict))

    # ---------- Plot Result ----------
    # # actual result
    # plt.figure()
    # # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.scatter(data_test[:, "spotMean"], data_test[:, "spotSTD"],
    #             c=label_test, cmap='viridis', marker='o', edgecolor='k', s=100)
    # plt.title('Actual Results')
    # # plt.xlabel(iris.feature_names[0])
    # # plt.ylabel(iris.feature_names[1])
    # plt.colorbar(ticks=[0, 1, 2], label='Classes')
    #
    # # predicted result
    # plt.subplot(1, 2, 2)
    # plt.scatter(data_test[:, "spotMean"], data_test[:, "spotSTD"], c=label_predict, cmap='viridis', marker='o', edgecolor='k', s=100)
    # plt.title('Predicted Results')
    # # plt.xlabel(iris.feature_names[0])
    # # plt.ylabel(iris.feature_names[1])
    # plt.colorbar(ticks=[0, 1, 2], label='Classes')
    #
    # plt.tight_layout()
    # plt.show()

import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import processData  # run processData.py
import processTest  # run processTest.py


# Main
if __name__ == '__main__':
    csv_path = "D:/srtp/knn/data/data.csv"

    # ---------- Load Train Data ----------
    full_csv = pd.read_csv(csv_path, index_col=0)
    # full_data include: "spotMean", "spotSTD", "speedMean", "speedSTD", "spotMax"
    full_data = full_csv.loc[:, "spotMean":"spotMax"]
    # full_data = full_csv.loc[:, "spotMean":"spotSTD"]
    full_label = full_csv.loc[:, "id"]

    data_train = full_data
    label_train = full_label
    # data_train, data_test, label_train, label_test \
    #     = train_test_split(full_data, full_label, test_size=0.3, random_state=None)

    # ---------- Fit Knn Model ----------
    # knn model name: knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data_train, label_train)

    # ---------- Load Test Data ----------
    csv_path = "D:/srtp/knn/data/test.csv"
    full_test = pd.read_csv(csv_path, index_col=0)
    # full_data include: "spotMean", "spotSTD", "speedMean", "speedSTD", "spotMax"
    data_test = full_test.loc[:, "spotMean":"spotMax"]
    # data_test = full_test.loc[:, "spotMean":"spotSTD"]
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
    # plt.scatter(data_test[:, "spotMean"], data_test[:, "spotSTD"], c=label_test, cmap='viridis', marker='o', edgecolor='k', s=100)
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

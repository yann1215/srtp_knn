import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier

import processData  # run processData.py


# Main
if __name__ == '__main__':
    # ---------- Load Train Data ----------
    train_path = "D:/srtp/knn/testPy/data.csv"
    data_train = pd.read_csv(train_path, usecols=["spotMean", "spotSTD", "speedMean", "speedSTD", "spotMax"])
    label_train = pd.read_csv(train_path, usecols=["id"])

    # ---------- Fit Knn Model ----------
    knn = KNeighborsClassifier(n_neighbors=3)  # knn model name: knn
    knn.fit(data_train, label_train)

    ec_test = [3.629381443,	0.323199987, 3.042405728, 2.093326917, 4.278350515]
    ec_predict = knn.predict(ec_test)
    print(ec_predict)

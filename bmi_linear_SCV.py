from sklearn import model_selection as cross_validation, svm, metrics
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv("labeled_data/LINE_v1.0_labeled_filter.embeddings", header=None, sep=' ', index_col=0)
# print(tbl.dtypes)
label = tbl[65]
del tbl[65]
# w = tbl["weight"] / 100
# h = tbl["height"] / 200
# dimensions_64 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
#                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
#                  44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]

# wh = pd.concat(dimensions_64, axis=1)

data_train, data_test, label_train, label_test = cross_validation.train_test_split(tbl, label)

print(label_test)

clf = svm.LinearSVC()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print(ac_score)
print(cl_report)
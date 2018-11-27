from sklearn import model_selection as cross_validation, svm, metrics
import matplotlib.pyplot as plt
import pandas as pd

data_matrix = pd.read_csv("labeled_data/node2vec_v1.0_labeled_filter.embeddings", header=None, sep=' ', index_col=0)
# print(tbl.dtypes)
label = data_matrix[65]
del data_matrix[65]

# wh = pd.concat(dimensions_64, axis=1)

data_train, data_test, label_train, label_test = cross_validation.train_test_split(data_matrix, label)

clf = svm.LinearSVC()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print(ac_score)
print(cl_report)
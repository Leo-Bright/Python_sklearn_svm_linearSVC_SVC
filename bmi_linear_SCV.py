from sklearn import model_selection as cross_validation, svm, metrics
import matplotlib.pyplot as plt
import pandas as pd

data_matrix = pd.read_csv("labeled_data/deepwalk_v1.0_labeled.embeddings", header=None, sep=' ', index_col=0)
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

# ===================================直接调用交叉验证评估模型==========================
clf2 = svm.LinearSVC()
scores = cross_validation.cross_val_score(clf2, data_matrix, label, cv=10, n_jobs=5) #cv为迭代次数。
cross_validation.cross_validate()
print(scores) # 打印输出每次迭代的度量值（准确度）
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) # 获取置信区间。（也就是均值和方差）
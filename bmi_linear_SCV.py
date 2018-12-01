from sklearn import model_selection as cross_validation, svm, metrics
import matplotlib.pyplot as plt
import pandas as pd

data_matrix = pd.read_csv("porto/labeled_emb/deepwalk/highway_64d_crossing.embeddings", header=None, sep=' ', index_col=0)
# print(tbl.dtypes)
rows_size, cols_size = data_matrix.shape
label = data_matrix[cols_size]
del data_matrix[cols_size]

# wh = pd.concat(dimensions_64, axis=1)

data_train, data_test, label_train, label_test = cross_validation.train_test_split(data_matrix, label)

clf = svm.LinearSVC()

clf.fit(data_train, label_train)
predict = clf.predict(data_test)
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print(ac_score)
print(cl_report)

f_result = open(r'porto/result/deepwalk/highway_64d_crossing.result', 'w+')
f_result.write(cl_report)

# ===================================直接调用交叉验证评估模型==========================
clf2 = svm.LinearSVC()
scores = cross_validation.cross_val_score(clf, data_matrix, label, cv=2, n_jobs=2) #cv为迭代次数。n_jobs为并发线程数
print(scores) # 打印输出每次迭代的度量值（准确度）
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) # 获取置信区间。（也就是均值和方差）
f_result.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

f_result.close()
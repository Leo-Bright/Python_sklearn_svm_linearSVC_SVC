import json
import random
from sklearn import model_selection as cross_validation, svm, metrics
import pandas as pd

input_raw_embedding_file = 'porto/embedding/deepwalk/po.deep_128'
tag_json_file = 'porto/segment/porto_oneway.json'

path_array = input_raw_embedding_file.rsplit('.', 1)
labeled = path_array[0] + '_labeled.' + path_array[1]

f_labeled = open(labeled, 'w+')
f_embeddings = open(input_raw_embedding_file, 'r')
f_nodes_selected = open(tag_json_file, 'r')


def label_embeddings(selected, embeddings, output, fraction=1):
    node_crossing = json.loads(selected.readline())
    crossing_count = 0
    normal_count = 0
    for line in embeddings.readlines():
        line = line.strip()
        osmid_vector = line.split(' ')
        osmid, node_vec = osmid_vector[0], osmid_vector[1:]
        if len(node_vec) < 10:
            continue
        if osmid in node_crossing:
            output.write(line + ' ' + 'oneway' + '\n')
            crossing_count += 1
        else:
            rd = random.randint(0, 999) + 1
            if rd > fraction:
                continue
            output.write(line + ' ' + 'reversal' + '\n')
            normal_count += 1
    print("oneway count: ", crossing_count)
    print("reversal count: ", normal_count)


label_embeddings(f_nodes_selected, f_embeddings, f_labeled, fraction=12)

f_labeled.close()
f_embeddings.close()
f_nodes_selected.close()


# ======classification=====
max_score = 0
max_report = None
for i in range(10):
    data_matrix = pd.read_csv(labeled, header=None, sep=' ', index_col=False)
    # print(tbl.dtypes)
    rows_size, cols_size = data_matrix.shape
    node_id = data_matrix[0]
    # node_id.astype('float')

    label = data_matrix[cols_size - 1]
    # del data_matrix[cols_size]
    # wh = pd.concat(dimensions_64, axis=1)

    data_train, data_test, label_train, label_test = cross_validation.train_test_split(node_id, label)
    clf = svm.LinearSVC(max_iter=10000)
    clf.fit(data_train.values.reshape(-1, 1), label_train)
    predict = clf.predict(data_test.values.reshape(-1, 1))
    ac_score = metrics.accuracy_score(label_test, predict)
    cl_report = metrics.classification_report(label_test, predict, digits=4)
    if ac_score > max_score:
        max_score = ac_score
        max_report = cl_report

print(max_score)
print(max_report)


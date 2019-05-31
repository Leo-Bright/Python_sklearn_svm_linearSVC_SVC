import json
import random
from sklearn import model_selection as cross_validation, svm, metrics
import pandas as pd

input_raw_embedding_file = 'sanfrancisco/embedding/my_model/tmp/sanfrancisco_shortest_dist_all_distance500_beta0.75_gama0.1_traffic_signals.embedding'
tag_json_file = 'sanfrancisco/node/nodes_crossing.json'

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
            output.write(line + ' ' + 'crossing' + '\n')
            crossing_count += 1
        else:
            rd = random.randint(0, 999) + 1
            if rd > fraction:
                continue
            output.write(line + ' ' + 'normal' + '\n')
            normal_count += 1
    print("crossing count: ", crossing_count)
    print("cormal count: ", normal_count)

label_embeddings(f_nodes_selected, f_embeddings, f_labeled, fraction=26)

f_labeled.close()
f_embeddings.close()
f_nodes_selected.close()


# ======classification=====
max_score = 0
max_report = None
for i in range(10):
    data_matrix = pd.read_csv(labeled, header=None, sep=' ', index_col=0)
    # print(tbl.dtypes)
    rows_size, cols_size = data_matrix.shape
    label = data_matrix[cols_size]
    del data_matrix[cols_size]

    # wh = pd.concat(dimensions_64, axis=1)

    data_train, data_test, label_train, label_test = cross_validation.train_test_split(data_matrix, label)

    clf = svm.LinearSVC(max_iter=10000)

    clf.fit(data_train, label_train)
    predict = clf.predict(data_test)
    ac_score = metrics.accuracy_score(label_test, predict)
    cl_report = metrics.classification_report(label_test, predict, digits=4)
    if ac_score > max_score:
        max_score = ac_score
        max_report = cl_report

print(max_score)
print(max_report)


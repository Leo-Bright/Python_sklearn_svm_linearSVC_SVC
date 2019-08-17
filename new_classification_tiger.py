import json
import random
from sklearn import model_selection as cross_validation, svm, metrics
import pandas as pd

input_raw_embedding_file = 'porto/embedding/my_model/pt_random_segment_distance500_type_classid_beta0.8.embedding'
tag_json_file = 'sanfrancisco/segment/sf_segments_tiger_nametype.json'

path_array = input_raw_embedding_file.rsplit('.', 1)
labeled = path_array[0] + '_labeled.' + path_array[1]
result_array = path_array[0].split('/', 2)
result_path = result_array[0] + '/result/' + result_array[2] + '.result'

f_labeled = open(labeled, 'w+')
f_embeddings = open(input_raw_embedding_file, 'r')
f_nodes_selected = open(tag_json_file, 'r')


def label_embeddings(selected, embeddings, output, keyset, fraction=1, ):
    data_selected_label = json.loads(selected.readline())
    positive_count = {k: 0 for k in keyset}
    negtive_count = 0
    for line in embeddings.readlines():
        line = line.strip()
        osmid_vector = line.split(' ')
        osmid, node_vec = osmid_vector[0], osmid_vector[1:]
        if len(node_vec) < 10:
            continue
        if osmid in data_selected_label:
            label = data_selected_label[osmid]
        if label is not None and label in keyset:
            output.write(line + ' ' + label + '\n')
            positive_count[label] += 1
        else:
            rd = random.randint(0, 999) + 1
            if rd > fraction:
                continue
            output.write(line + ' ' + 'unknown' + '\n')
            negtive_count += 1
    print("positive count: ", positive_count)
    print("negtive count: ", negtive_count)


label_embeddings(f_nodes_selected, f_embeddings, f_labeled, keyset=('St', 'Ave',) , fraction=300, )

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

with open(result_path, 'w+') as f:
    f.write(max_report)


import json
import random


f_labeled = open(r'labeled_data/node2vec_v1.0_labeled_filter_2.embeddings', 'w+')
f_embeddings = open(r'data/node2vec_highway_64d_v1.0.embeddings', 'r')
f_nodes_selected = open(r'data/nodes_crossing.json', 'r')


def label_embeddings(selected, embeddings, output, fraction=10):
    node_crossing = json.loads(selected.readline())
    crossing_count = 0
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
            rd = random.randint(0, 99) + 1
            if rd > fraction:
                continue
            output.write(line + ' ' + 'normal' + '\n')
    print("crossing count: ", crossing_count)

label_embeddings(f_nodes_selected, f_embeddings, f_labeled, fraction=2)

f_labeled.close()
f_embeddings.close()
f_nodes_selected.close()

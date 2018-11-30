import json
import random


f_labeled = open(r'labeled_data/node2vec_v1.0_labeled_1000.embeddings', 'w+')
f_embeddings = open(r'data/node2vec_highway_64d_v1.0.embeddings', 'r')
f_nodes_selected = open(r'data/nodes_crossing.json', 'r')


def label_embeddings(selected, embeddings, output, positive_sample=10000, other_sample=10000, fraction=1):
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
            if crossing_count < positive_sample:
                output.write(line + ' ' + 'crossing' + '\n')
                crossing_count += 1
        else:
            if normal_count < other_sample:
                rd = random.randint(0, 99) + 1
                if rd > fraction:
                    continue
                output.write(line + ' ' + 'normal' + '\n')
                normal_count += 1
        if crossing_count == positive_sample and normal_count == other_sample:
            break
    print("crossing count: ", crossing_count)


label_embeddings(f_nodes_selected, f_embeddings, f_labeled, positive_sample=1000, other_sample=1000 ,fraction=1)

f_labeled.close()
f_embeddings.close()
f_nodes_selected.close()

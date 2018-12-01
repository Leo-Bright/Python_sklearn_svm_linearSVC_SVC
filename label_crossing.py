import json
import random


f_labeled = open(r'porto/labeled_emb/deepwalk/highway_64d_crossing.embeddings', 'w+')
f_embeddings = open(r'porto/embedding/deepwalk/deepwalk_highway_64d.embeddings.txt', 'r')
f_nodes_selected = open(r'porto/node/nodes_crossing.json', 'r')


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

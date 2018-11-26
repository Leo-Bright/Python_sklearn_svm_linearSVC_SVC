import json


f_labeled = open(r'data/deepwalk_v1.0_labeled.embeddings', 'w+')
f_embeddings = open(r'data/deepwalk_highway_allNodes_64d_v1.0.embeddings', 'r')
f_nodes_selected = open(r'data/nodes_crossing.json', 'r')


def label_embeddings(selected, embeddings, output):
    node_crossing = json.loads(selected.readline())
    crossing_count = 0
    for line in embeddings.readlines():
        line = line.strip()
        osmid_vector = line.split(' ')
        osmid, node_vec = osmid_vector[0], osmid_vector[1:]
        if len(node_vec) < 10:
            continue
        if osmid in node_crossing:
            print('label a crossing node')
            output.write(line + ' ' + 'crossing' + '\n')
            crossing_count += 1
        else:
            print('this is not a crossing')
            output.write(line + ' ' + 'normal' + '\n')
    print(crossing_count)

label_embeddings(f_nodes_selected, f_embeddings, f_labeled)

f_labeled.close()
f_embeddings.close()
f_nodes_selected.close()

import json
import random


f_labeled = open(r'sanfrancisco/labeled_emb/deepwalk/sf_labeled.embedding64', 'w+')
f_embeddings = open(r'sanfrancisco/embedding/deepwalk/sf.embedding64', 'r')
f_intersection_nodes = open(r'sanfrancisco/node/nodes_intersection.json', 'r')


def label_embeddings(selected, embeddings, output, fraction=10, index=(2, 3, 4)):
    node_intersection = json.loads(selected.readline())
    intersect = 0
    intersect_2 = 0
    intersect_3 = 0
    intersect_4 = 0
    normal = 0
    for line in embeddings.readlines():
        line = line.strip()
        osmid_vector = line.split(' ')
        osmid, node_vec = osmid_vector[0], osmid_vector[1:]
        if len(node_vec) < 10:
            continue
        if 2 in index:
            node_intersection_2 = node_intersection['2']
            if osmid in node_intersection_2:
                output.write(line + ' ' + 'intersect_2' + '\n')
                intersect_2 += 1
                intersect += 1
                continue
        if 3 in index:
            node_intersection_3 = node_intersection['3']
            if osmid in node_intersection_3:
                output.write(line + ' ' + 'intersect_3' + '\n')
                intersect_3 += 1
                intersect += 1
                continue
        if 4 in index:
            node_intersection_4 = node_intersection['4']
            if osmid in node_intersection_4:
                output.write(line + ' ' + 'intersect_4' + '\n')
                intersect_4 += 1
                intersect += 1
                continue
        rd = random.randint(0, 999) + 1
        if rd > fraction:
            continue
        normal += 1
        output.write(line + ' ' + 'normal' + '\n')
    print("intersection 2 :" + str(intersect_2))
    print("intersection 3 :" + str(intersect_3))
    print("intersection 4 :" + str(intersect_4))
    print("intersection 4 :" + str(normal))


label_embeddings(f_intersection_nodes, f_embeddings, f_labeled, fraction=50, index=(2, ))

f_labeled.close()
f_embeddings.close()
f_intersection_nodes.close()

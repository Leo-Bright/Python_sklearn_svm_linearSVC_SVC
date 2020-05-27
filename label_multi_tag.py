def label_embeddings(embeddings, output, node2tag):

    tag_conut = {}

    node2emb = {}

    emb_count = 0

    with open(embeddings) as f:
        for line in f:
            osmid_vector = line.strip().split()
            osmid, node_vec = osmid_vector[0], osmid_vector[1:]
            if len(node_vec) < 3:
                continue
            node2emb[osmid] = node_vec
            emb_count += 1
    print(emb_count)

    with open(output, 'w+') as f:
        for key in node2emb:
            emb = node2emb[key]
            if key not in node2tag:
                continue
            tag = node2tag[key]
            f.write(key + ' %s' % ' '.join(emb) + ' ' + str(tag) + '\n')
            if str(tag) not in tag_conut:
                tag_conut[str(tag)] = 0
            tag_conut[str(tag)] += 1

    for key in tag_conut:
        print(key + " count: ", str(tag_conut[key]))


def gen_node_tag_dict(tag_file_path):

    node2tag = {}

    with open(tag_file_path) as f:
        for line in f:
            osmid_tag = line.strip().split(' ')
            osmid = osmid_tag[0]
            tag = osmid_tag[1]
            node2tag[osmid] = int(tag)

    return node2tag


if __name__ == '__main__':

    node_tag_dict = gen_node_tag_dict('sanfrancisco/node/sanfrancisco_nodes_with_all.tag')

    src_emb_file_path = 'sanfrancisco/embedding/my_model/sanfrancisco_shortest_dist_tag_type_window_width500_crossing.embedding'
    labeled_emb_path = 'sanfrancisco/embedding/my_model/sanfrancisco_shortest_dist_tag_type_window_width500_crossing_labeled_all_tag.embedding'

    label_embeddings(src_emb_file_path, labeled_emb_path, node_tag_dict)

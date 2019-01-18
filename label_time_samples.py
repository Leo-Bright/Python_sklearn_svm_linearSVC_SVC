import numpy as np
from numpy import float64, float16


def main(time_samples, embeddings, output, method):

    output_file = open(output, 'w+')

    osmid_embeddings = {}

    with open(embeddings, 'r') as embeddings_file:
        for line in embeddings_file:
            line = line.strip()
            osmid_vector = line.split(' ')
            osmid, node_vec = osmid_vector[0], osmid_vector[1:]
            if len(node_vec) < 10:
                continue
            osmid_embeddings[osmid] = node_vec

    with open(time_samples, 'r') as time_samples_file:
        for line in time_samples_file:
            line = line.strip()
            node_sequence_time = line.split(' ')
            node_sequence = node_sequence_time[:-1]
            travel_time = node_sequence_time[-1]
            nodes_embeddings = []
            for node in node_sequence:
                if node not in osmid_embeddings:
                    continue
                nodes_embeddings.append(osmid_embeddings[node])

            result = combine_embeddings(nodes_embeddings, method)
            output_file.write('%s\n' % ' '.join(map(str, result)))

    output_file.close()


def combine_embeddings(embeddings_list, method):
    matrix = np.array(embeddings_list, dtype=float16)
    if method == '+':
        result = matrix.sum(axis=0)
    elif method == '*':
        matrix = np.abs(matrix)
        matrix = np.log(matrix)
        matrix = np.abs(matrix)
        result = matrix.sum(axis=0)
    else:
        col_size = matrix.shape[1]
        result = matrix.sum(axis=0)/col_size
    print(result.tolist())
    return result.tolist()


main(time_samples='sanfrancisco/node/sf_travel_time_21.samples',
     embeddings='sanfrancisco/embedding/my_model/sanfrancisco_shortest_wn160_d128_ns5_ws5.embeddings',
     output='sanfrancisco/labeled_emb/my_model/sanfrancisco_shortest_wn160_d128_ns5_ws5_time_21.embeddings',
     method='*')

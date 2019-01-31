import numpy as np
from numpy import float16


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

    print(output.rsplit('/', 1)[1])
    with open(time_samples, 'r') as time_samples_file:
        line_count = 0
        # total_output_size = 1000000
        for line in time_samples_file:
            # if line_count >= total_output_size:
            #     break
            if line_count % 100000 == 0:
                print('process trajectory: ', line_count)
            line_count += 1
            line = line.strip()
            node_sequence_time = line.split(' ')
            if len(node_sequence_time) < 5:
                continue
            node_sequence = node_sequence_time[:-1]
            travel_time = node_sequence_time[-1]
            if int(travel_time) < 10 or int(travel_time) > 1000:
                continue
            nodes_embeddings = []
            for node in node_sequence:
                if node not in osmid_embeddings:
                    continue
                nodes_embeddings.append(osmid_embeddings[node])

            result = combine_embeddings(nodes_embeddings, method)
            output_file.write('%s\n' % ' '.join(map(str, result + [travel_time])))

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
    elif method == '&':
        result = np.append(matrix[0], matrix[-1])
    elif method == '-':
        col_size = matrix.shape[0]
        result = matrix.sum(axis=0)/col_size
    else:
        raise Exception
    return result.tolist()


main(time_samples='sanfrancisco/node/sf_trajectory_node_travel_time_450.travel',
     embeddings='sanfrancisco/embedding/deepwalk/sf.embedding128',
     output='sanfrancisco/labeled_emb/deepwalk/sf_deepwalk_time_450_multi.embeddings',
     method='*')

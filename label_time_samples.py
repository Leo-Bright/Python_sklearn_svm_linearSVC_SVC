def main(time_samples, embeddings, output):

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
            [start, end, travel_time] = line.split(' ')
            start_emb = osmid_embeddings[start] if start in osmid_embeddings else None
            end_emb = osmid_embeddings[end] if end in  osmid_embeddings else None
            if not start_emb or not end_emb:
                continue
            result = [start, end] + start_emb + end_emb + [travel_time]
            output_file.write('%s\n' % ' '.join(map(str, result)))

    output_file.close()


main(time_samples='sanfrancisco/node/sf_travel_time_7.samples',
     embeddings='sanfrancisco/embedding/my_model/sanfrancisco_shortest_wn160_d128_ns5_ws5.embeddings',
     output='sanfrancisco/labeled_emb/my_model/sanfrancisco_shortest_wn160_d128_ns5_ws5_time_7.embeddings',)

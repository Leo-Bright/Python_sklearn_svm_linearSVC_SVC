embedding_input_files = ["sanfrancisco/embedding/line/highway_allNodes_LINE.embeddings",
                   "sanfrancisco/embedding/line/highway_allNodes_LINE2.embeddings",
                   ]
embedding_output_file = "sanfrancisco/embedding/line/highway_allNodes_LINE3.embeddings"

embeddings_result = {}  # {osmid:[embeddings...]}
for embedding_file in embedding_input_files:
    with open(embedding_file, 'r') as file:
        for line in file:
            items = line.strip().split(' ')
            if len(items) <= 2:
                continue
            osmid = items[0]
            embeddings = items[1:]
            if osmid not in embeddings_result:
                embeddings_result[osmid] = embeddings
            else:
                embeddings_result[osmid] += embeddings

with open(embedding_output_file, 'w+') as file:
    for osmid, embeddings in embeddings_result.items():
        file.write(osmid + ' ')
        for index, item in enumerate(embeddings):
            if index != len(embeddings) - 1:
                file.write(item + ' ')
                continue
            file.write(item + '\n')


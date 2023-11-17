# This is a sample Python script.
import uuid
import os
import jsonlines
from fastembed.embedding import FlagEmbedding as Embedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

data = "confluence"
collection_name = data

def generate_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_all_file_names(path: str)->list[str]:
    return os.listdir(path)

if __name__ == '__main__':
    client = QdrantClient(host="localhost", port=6333)
    first_collection = client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )

    embedding_model = Embedding(model_name="intfloat/multilingual-e5-large", max_length=512)

    chunk_number = 0
    file_names = get_all_file_names("./data/Confluence-Export-txt")
    for file_name in file_names:
        handel = open(f'./data/Confluence-Export-txt/{file_name}', 'r')
        lines = handel.readlines()
        chunks = generate_chunks(lines, 10)

        for chunk in chunks:
            chunk_number = chunk_number + 1
            print(f'processing chunk #{chunk_number}')

            embeddings = [e.tolist() for e in embedding_model.passage_embed(chunk)]
            ids = [uuid.uuid4().__str__() for i in range(len(embeddings))]
            payloads = [{'text': p, 'file_name': file_name} for p in chunk]

            client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                )
            )





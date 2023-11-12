# This is a sample Python script.
import uuid

import jsonlines
from fastembed.embedding import FlagEmbedding as Embedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

data = "dune"
collection_name = data

def generate_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    dune = open(f'data/{data}.txt', 'r')
    lines = dune.readlines()

    chunks = generate_chunks(lines, 10)

    client = QdrantClient(host="localhost", port=6333)
    first_collection = client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

    embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512)

    chunk_number = 0
    for chunk in chunks:
        chunk_number = chunk_number + 1
        print(f'processing chunk #{chunk_number}')

        embeddings = [e.tolist() for e in embedding_model.passage_embed(chunk)]
        ids = [uuid.uuid4().__str__() for i in range(len(embeddings))]
        payloads = [{'text': p} for p in chunk]

        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=payloads
            )
        )





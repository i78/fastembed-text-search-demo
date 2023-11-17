# This is a sample Python script.
import uuid

import jsonlines
from fastembed.embedding import FlagEmbedding as Embedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

collection_name = "confluence"

if __name__ == '__main__':
    client = QdrantClient(host="localhost", port=6333)
    embedding_model = Embedding(model_name="intfloat/multilingual-e5-large", max_length=512)

    # query = "query: Was ist die maximale Mitgliedsgebühr für ein Fitnessstudio?"
    # query = "query: Ich möchte einen Firmenwagen. An wen muss ich mich wenden?"
    # query = "query: Wann sollte ich Logstash verwenden?"
    # query = "query: Darf ich in der Schweiz arbeiten?"

    query = "query: Kann ich einen Dienstwagen haben?"

    query_vector = [e.tolist() for e in embedding_model.embed(query)],
    # query_vector = embedding_model.embed(query),

    # print(query_vector)


    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector[0][0],
        limit=8,
        with_payload=["text", "file_name"],
        append_payload=True
    )

    print(search_result)

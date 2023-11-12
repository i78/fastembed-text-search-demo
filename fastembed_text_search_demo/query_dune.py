# This is a sample Python script.
import uuid

import jsonlines
from fastembed.embedding import FlagEmbedding as Embedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

data = "dune"
collection_name = data

with jsonlines.open("data/documents.json") as documents_file:
    documents = [row for row in documents_file]



if __name__ == '__main__':
    client = QdrantClient(host="localhost", port=6333)
    embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512)

    #query = "query: Which religion is used in the text?"
    #query = "query: Whats the name of the imperial planetologist?"
    # Interessant, weil nicht im Buch und er aber was zur Größe findet
    #query = "query: What is the weight of a sandworm?"
    # Spoiler: Es gibt keine, aber er stellt die Nähe zu Aussagen zur Familie her
    # query = "query: Who is the third wife of Paul Atreides?"
    # query = "query: What is the meaning of water on Arrakis?"
    #query = "query: Do they sell drinks on the beach?"
    #query = "query: Supposed the Fremen lived on Giedi Prime, how would they make money?"
    #query = "query: What is the name of Leto Atreides son and what is the most severe danger in the deserts?"
    #query = "query: Which elite unit comes from the same planet as the author of the chronicles of Muad'dib?"
    query = "query: Who is the author of the chronicles of Muad'dib?"

    query_vector = [e.tolist() for e in embedding_model.embed(query)],
    # query_vector = embedding_model.embed(query),

    # print(query_vector)


    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector[0][0],
        limit=8,
        with_payload= ["text"],
        append_payload=True
    )

    print(search_result)

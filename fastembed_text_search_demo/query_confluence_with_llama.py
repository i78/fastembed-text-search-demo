# This is a sample Python script.
import uuid

import jsonlines
from fastembed.embedding import FlagEmbedding as Embedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

collection_name = "confluence"

if __name__ == '__main__':
    client = QdrantClient(host="localhost", port=6333)
    embedding_model = Embedding(model_name="intfloat/multilingual-e5-large", max_length=512)

    query = "Was ist die maximale monatliche Mitgliedsgebühr für ein Fitnessstudio die übernommen wird?"
    # query = "Ich möchte einen Firmenwagen. An wen muss ich mich wenden?"
    # query = "Wann sollte ich Logstash verwenden?"
    # query = "Darf ich in der Schweiz arbeiten?"

    # query = "Kann ich einen Dienstwagen haben?"

    query_vector = [e.tolist() for e in embedding_model.embed(f"query: {query}")],
    # query_vector = embedding_model.embed(query),

    # print(query_vector)


    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector[0][0],
        limit=8,
        with_payload=["text", "file_name"],
        append_payload=True
    )

    # model_path = 'openlm-research/open_llama_3b'
    # model_path = 'openlm-research/open_llama_7b'
    model_path = 'openlm-research/open_llama_13b'

    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto', offload_folder='offline'
    )

    texts = [result.payload.get("text") for result in search_result]
    searchInformation = '\n'.join([str(text) for text in texts])
    prompt = f'<<SYS>>You are a helpful research assistant. The following functions, if any, are available for you to fetch further data to answer user questions: {searchInformation}<</SYS>>\n[INST]Question: {query}[/INST]'
    # prompt = f'<<<SYS>>>Use the following information and answer the question in german.  \n {searchInformation} <<</SYS>>> \n\n [INST]Question: {query}[/INST]'

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to('mps')

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=100
    )
    print(tokenizer.decode(generation_output[0]))

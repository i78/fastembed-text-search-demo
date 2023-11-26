import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

if __name__ == '__main__':
    model_path = 'openlm-research/open_llama_3b'
    # model_path = 'openlm-research/open_llama_7b'
    # model_path = 'openlm-research/open_llama_13b'

    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto', offload_folder='offline'
    )

    prompt = 'Q: What is the largest animal?\nA:'
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to('mps')

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=32
    )
    print(tokenizer.decode(generation_output[0]))
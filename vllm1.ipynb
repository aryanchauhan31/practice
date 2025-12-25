from vllm import LLM, SamplingParams
import time

llm = LLM(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    quantization="awq",
    dtype="half",
    max_model_len=4096,
    gpu_memory_utilization=0.90
)
tokenizer = llm.get_tokenizer()
messages = [
    {"role": "user", "content": "Write an essay on Delhi in 200 words."}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)


sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
start = time.time()
outputs = llm.generate([formatted_prompt], sampling_params)
end = time.time()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
duration = end - start
throughput = total_tokens / duration

print(f"Total Tokens Generated: {total_tokens}")
print(f"Time Taken: {duration:.2f}s")
print(f"Decoding Throughput: {throughput:.2f} tokens/s")




// Adding requests: 100%
//  1/1 [00:00<00:00, 99.68it/s]
// Processed prompts: 100%
//  1/1 [00:12<00:00, 12.56s/it, est. speed input: 3.66 toks/s, output: 20.22 toks/s]
// Total Tokens Generated: 254
// Time Taken: 12.57s
// Decoding Throughput: 20.20 tokens/s

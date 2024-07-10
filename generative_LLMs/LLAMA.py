import transformers
import torch

model_id = "alokabhishek/Meta-Llama-3-8B-Instruct-bnb-4bit"

print('Making pipeline...')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt_instruction = "Extract all the entities (ORG, LOC, PER, MISC) in the following piece of text."
user_prompt = """
Serbia needs to start work on ending its moratorium on nuclear energy as it faces rising electricity consumption which can only be tackled by the construction of large and small nuclear power plants, president Aleksandar Vucic said.
Vucic called for work to start on changing regulations related to new nuclear plants and to end the moratorium, which has been in force since 1989.
“I just want you to know that by 2050 we will be consuming four times more electricity than today,” Vucic was quoted as saying at a government session on 4 April.
“No matter what we do, no matter how we do it, we don’t stand a chance if we don’t start addressing that problem quickly.
“And solving that problem is only possible by building large and small nuclear power plants,” Vucic said, according to local press reports.
"""

chat_messages = [
            {"role": "system", "content": str(prompt_instruction)},
            {"role": "user", "content": str(user_prompt)},
        ]

print('Making prompt...')
prompt = pipeline.tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

print('Making output...')
output = pipeline(
    prompt,
    do_sample=True,
    max_new_tokens=1024,
    temperature=1,
    top_k=50,
    top_p=1,
    num_return_sequences=1,
    pad_token_id=pipeline.tokenizer.pad_token_id,
    eos_token_id=terminators,
)


print(output[0]["generated_text"][len(prompt):])
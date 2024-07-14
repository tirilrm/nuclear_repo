import transformers
import torch
import time

start = time.time()
print('Cuda version:', torch.version.cuda)

model_id = "alokabhishek/Meta-Llama-3-8B-Instruct-bnb-4bit"

print('Making pipeline...')
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float32},
    device_map="auto",
)

prompt_instruction = """
Task: Named Entity Recognition (NER)
Identify entities in the text and provide the following information for each entity in a dictionary format:
	'entity': the type of entity (using codes ORG, LOC, PER, MISC).
	'score': confidence level for the identification;
	'word': the entity text;
	'start': start character position of the entity in the text
	'end': end character position of the entity in the text

Entity labeling instructions:
	Mark the first word of an entity with 'B-' prefix and subsequent words of the same entity with the 'I-' prefix.
	For single-word entities, use B- prefix as it is the beginning of the entity.
	Ensure the results are sorted by their appearance in the text.
	'start' and 'end' indicate position in given paragraph (the text can contain multiple paragraphs).

Example output:
Return results on the form:
    {'entity': 'B-LOC', 'score': 0.9985879, 'index': 19, 'word': 'New', 'start': 94, 'end': 96},
    {'entity': 'I-LOC', 'score': 0.9958712, 'index': 25, 'word': 'York', 'start': 98, 'end': 101},
	etc...
"""

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


end = time.time()

print(f'Time taken: {(end-start)/60} minutes')

result = output[0]["generated_text"][len(prompt):]
print(result)

time_taken = (end-start)/60

# Corrected file_name generation
file_name = "output_" + str(int(time.time())) + ".txt"

with open(file_name, "w") as file:
	file.write(f"TIME TAKEN:\n {time_taken:>2f} minutes\n")
	file.write(f"\nPROMPT INSTRUCTION:")
	file.write(prompt_instruction)
	file.write("\nUSER PROMPT:")
	file.write(user_prompt)
	file.write("\nLLAMA OUTPUT:\n")
	file.write(result)
	file.write("\n")
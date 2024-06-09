from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load a compatible model for question answering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Tokenize the input data
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    return tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384)

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

# Create DataLoader for the evaluation dataset
from torch.utils.data import DataLoader

eval_dataset = tokenized_squad["validation"]
eval_dataloader = DataLoader(eval_dataset, batch_size=16)

# Evaluate the model
model.eval()
all_predictions = []
for batch in eval_dataloader:
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Extract answers
        batch_predictions = extract_answers(start_logits, end_logits, batch, tokenizer)
        all_predictions.extend(batch_predictions)

# Post-process and evaluate using SQuAD script
def extract_answers(start_logits, end_logits, batch, tokenizer):
    answers = []
    for i in range(len(start_logits)):
        start_idx = torch.argmax(start_logits[i])
        end_idx = torch.argmax(end_logits[i])
        if start_idx <= end_idx:
            answer = tokenizer.decode(batch['input_ids'][i][start_idx:end_idx+1], skip_special_tokens=True)
        else:
            answer = ""
        answers.append(answer)
    return answers

import json

predictions = {example["id"]: answer for example, answer in zip(squad["validation"], all_predictions)}
with open("predictions.json", "w") as f:
    json.dump(predictions, f)

# Run the evaluation script
import subprocess
subprocess.run(["python", "evaluate-v2.0.py", "squad/dev-v2.0.json", "predictions.json"])

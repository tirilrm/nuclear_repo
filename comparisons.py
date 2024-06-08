from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_datasets


model_names = {
    "BERT": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa": "deepset/roberta-base-squad2",
    "GPT-2": "gpt2",
    "T5": "t5-large"
}

tokenizers = {name: AutoTokenizer.from_pretrained(model) for name, model in model_names.items()}
models = {name: AutoModelForQuestionAnswering.from_pretrained(model) for name, model in model_names.items()}

squad = load_datasets("datasets/squad.json")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import difflib
import time
import json

models = {
    'BERT': 'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad',
    'DistilBERT': 'distilbert/distilbert-base-cased-distilled-squad',
    'RoBERTa': 'deepset/roberta-base-squad2',
    #'TinyLLAMA': 'TinyLlama/TinyLlama-1.1B-step-50K-105b',
    #'Mistral': 'mistralai/Mistral-7B-v0.1'
}

squad = load_dataset('squad')

def is_close_match(predicted, gold_answers, threshold=0.99):
    for gold in gold_answers:
        similarity = difflib.SequenceMatcher(None, predicted, gold).ratio()
        if similarity >= threshold:
            return True
    return False

results = {name: {} for name in models}
for name in models:
    print('Currently testing:', name)

    model_name = models[name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

    correct = 0
    total = 0
    start = time.time()
    for example in squad['validation']:
        question = example['question']
        context = example['context']
        result = qa_pipeline(question=question, context=context)
        pred = result['answer'].strip().lower()
        gold = [g.strip().lower() for g in example['answers']['text']]

        if is_close_match(pred, gold):
            correct += 1
        total += 1
    
    end = time.time()
    accuracy = correct/total
    length = end-start

    results[name]['Source'] = model_name
    results[name]['Accuracy'] = accuracy
    results[name]['Time'] = length

with open('squad_results.json', 'w') as f:
    json.dump(results, f, indent=4)

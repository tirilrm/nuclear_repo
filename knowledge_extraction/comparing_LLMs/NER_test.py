from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import time
import json

from _NER import join_tokens, merge_result, find_word_indices, get_predicted_tags, calculate_metrics

wikiann = load_dataset('wikiann', 'en')

models = {
    'BERT': 'dslim/bert-base-NER',
    'DistilBERT': 'dslim/distilbert-NER',
    'RoBERTa1': 'MMG/roberta-base-ner-english',
    'RoBERTa2': '51la5/roberta-large-NER',
    'RoBERTa3': 'FacebookAI/xlm-roberta-large-finetuned-conll03-english',
    'ALBERT1': 'ArBert/albert-base-v2-finetuned-ner',
    'ALBERT2': 'Jorgeutd/albert-base-v2-finetuned-ner'
}

def test_model(model_name, wiki_dataset, length=-1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipeline = pipeline('ner', model=model_name, tokenizer=tokenizer)
    
    start_time = time.time()

    tokens_all = wiki_dataset['tokens']
    tags_all = wiki_dataset['ner_tags']

    iterations = len(tokens_all)
    if length > 0:
        iterations = length

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(iterations):
        tokens = tokens_all[i]
        true_tags = tags_all[i]
        text = join_tokens(tokens)
        result = ner_pipeline(text)
        merged_result = merge_result(result, model_name)
        predicted_tags = get_predicted_tags(merged_result, tokens)

        tp, fp, tn, fn = calculate_metrics(predicted_tags, true_tags)

        TP += tp
        FP += fp
        TN += tn
        FN += fn

        if i%100 == 0:
            print('Iteration', i)

    end_time = time.time()
    duration = end_time - start_time

    accuracy = TP/(TP + FP + TN + FN) if (TP + FP + TN + FN) else 0
    precision = TP/(TP + FP) if (TP + FP) > 0 else 0
    recall = TP/(TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return duration, accuracy, precision, recall, f1_score

data = wikiann['test'][:]
results = {name: {} for name in models}

time_start = time.time()

for name in models:
    model_name = models[name]
    print('Currently testing:', model_name)
    duration, accuracy, precision, recall, f1_score = test_model(model_name, data)
    results[name]['Source'] = model_name
    results[name]['Accuracy'] = accuracy
    results[name]['Precision'] = precision
    results[name]['Recall'] = recall
    results[name]['F1 Score'] = f1_score
    results[name]['Time'] = duration

time_end = time.time()
print(f'Testing took {(time_end-time_start)/60:.2f} minutes')

with open('results/240630_wikiall_results.json', 'w') as f:
    json.dump(results, f, indent=4)
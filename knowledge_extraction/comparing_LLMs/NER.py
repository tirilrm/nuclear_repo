from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import time
import json

wikiann = load_dataset('wikiann', 'en')

models = {
    'BERT': 'dslim/bert-base-NER',
    'DistilBERT': 'dslim/distilbert-NER',
    'RoBERTa (large)': 'Jean-Baptiste/roberta-large-ner-english',
    'RoBERTa (small)': 'MMG/roberta-base-ner-english',
    'RoBERTa (large2)' : 'FacebookAI/xlm-roberta-large-finetuned-conll03-english',
    'AlBERT': 'ArBert/albert-base-v2-finetuned-ner'
}

def join_tokens(tokens):
    return ' '.join(tokens)

label_map = {
    '0': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    "B-LOC": 5,
    'I-LOC': 6,

    'PER' : 1,
    'ORG': 3,
    'LOC': 5
    
}

def merge_result(entities):
    merged_entities = []
    current = None

    for entity in entities:
        if current == None:
            current = entity
        else:
            if entity['word'].startswith('##'):
                current['word'] += entity['word'][2:]
                current['end'] = entity['end']
                current['score'] = min(current['score'], entity['score'])
            else:
                merged_entities.append(current)
                current = entity
    
    if current is not None:
        merged_entities.append(current)
    
    return merged_entities

def find_word_indices(word, tokens):
    indices = []
    for i, token in enumerate(tokens):
        if token == word:
            indices.append(i)
    return indices

def get_predicted_tags(results, tokens):
    predicted_tags = [0 for _ in range(len(tokens))]
    for result in results:
        entity = result['entity']
        if entity in label_map.keys():
            word = result['word']
            indices = find_word_indices(word, tokens)
            for index in indices:
                predicted_tags[index] = label_map[entity]
    
    return predicted_tags

def calculate_metrics(tags_pred, tags_gold):
    tp, fp, tn, fn = 0, 0, 0, 0

    for pred, gold in zip(tags_pred, tags_gold):
        if pred == gold and gold != '0':
            tp += 1
        elif gold != '0' and pred != gold:
            fn += 1
        elif gold == '0' and pred != '0':
            fp += 1
        elif gold == '0' and pred == '0':
            tn += 1
    
    return tp, fp, tn, fn

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
        merged_result = merge_result(result)
        predicted_tags = get_predicted_tags(merged_result, tokens)

        tp, fp, tn, fn = calculate_metrics(predicted_tags, true_tags)

        TP += tp
        FP += fp
        TN += tn
        FN += fn

    end_time = time.time()
    duration = end_time - start_time

    accuracy = TP/(TP + FP + TN + FN) if (TP + FP + TN + FN) else 0
    precision = TP/(TP + FP) if (TP + FP) > 0 else 0
    recall = TP/(TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return duration, accuracy, precision, recall, f1_score

data = wikiann['test'][:]
results = {name: {} for name in models}

for name in models:
    print('Currently testing:', name)
    model_name = models[name]
    duration, accuracy, precision, recall, f1_score = test_model(model_name, data)
    results[name]['Source'] = model_name
    results[name]['Accuracy'] = accuracy
    results[name]['Precision'] = precision
    results[name]['Recall'] = recall
    results[name]['F1 Score'] = f1_score
    results[name]['Time'] = duration

with open('results/240623_wikiall_results2.json', 'w') as f:
    json.dump(results, f, indent=4)

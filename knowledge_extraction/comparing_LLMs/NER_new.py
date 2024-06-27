from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import time
import json

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

label_map = {
    '0': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,

    'PER' : 1,
    'ORG': 3,
    'LOC': 5
}

reverse_label_map = {
    '0': '0',
    '1': 'B-PER',
    '2': 'I-PER',
    '3': 'B-ORG',
    '4': 'I-ORG',
    '5': 'B-LOC',
    '6': 'I-LOC',
    '7': 'B-MISC',
    '8': 'I-MISC'
}

def join_tokens(tokens):
    return ' '.join(tokens)

def merge_result(entities, name):
    merged_entities = []
    current = None

    if name in ['dslim/bert-base-NER', 'dslim/distilbert-NER']:
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
    else: 
        symbol = '▁'
        if name in ['MMG/roberta-base-ner-english']:
            symbol = 'Ġ'

        for entity in entities:
            if current == None:
                current = entity
            else:
                if not entity['word'].startswith(symbol):
                    current['word'] += entity['word']
                    current['end'] = entity['end']
                    current['score'] = min(current['score'], entity['score'])
                else:
                    current['word'] = current['word'][1:]
                    merged_entities.append(current)
                    current = entity
        if current is not None:
            current['word'] = current['word'][1:]
            merged_entities.append(current)
    return merged_entities

def find_word_indices(word, tokens):
    indices = []
    for i, token in enumerate(tokens):
        if token.lower().strip() == word.lower().strip():
            indices.append(i)
    return indices

def get_predicted_tags(results, tokens):
    predicted_tags = [0 for _ in range(len(tokens))]
    for result in results:
        entity = result['entity']
        if 'LABEL' in entity:
            entity = reverse_label_map[entity[-1]]
        if entity in label_map.keys(): # Ignore miscellaneous tags (not labeled in wikidata)
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

with open('results/240626_wikiall_results.json', 'w') as f:
    json.dump(results, f, indent=4)
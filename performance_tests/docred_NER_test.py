from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd
import itertools
from difflib import SequenceMatcher
import json
import numpy as np

import importlib
import _NER
importlib.reload(_NER)
from _NER import merge_result, combine_entities

def get_info(docred_instance):
    title = docred_instance['title']

    flattened_sents = [' '.join(sublist) for sublist in docred_instance['sents']]
    text = '\n'.join(flattened_sents)

    nested_entities = docred_instance['vertexSet']
    entities = list(itertools.chain(*nested_entities))
    
    head = docred_instance['labels']['head']
    tail = docred_instance['labels']['tail']
    r_id = docred_instance['labels']['relation_id']
    r_text = docred_instance['labels']['relation_text']
    evidence = docred_instance['labels']['evidence']

    return title, text, entities, head, tail, r_id, r_text, evidence

def has_approximate_match(gold, preds, threshold=0.8):
    gold_name, gold_type = gold

    # Note: removed requirement that entity type must match
    for pred_name, pred_type in preds: 
        ratio = SequenceMatcher(None, gold_name, pred_name).ratio()
        if ratio >= threshold:
            return (pred_name, pred_type)
    
    return None

data = load_dataset('docred', trust_remote_code=True)
datasets = [
    ('train_annotated', 3035),
    ('train_distant', 101873)
]

model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_pipeline = pipeline('ner', model=model_name, tokenizer=tokenizer)

N = 3000
mismatches = []
approx_matches = []
results_all = {}
results_all['N'] = N

correct_all = 0
total_all = 0
mismatch_count_all = 0
approx_match_count_all = 0
exact_match_count_all = 0

for d, nrows in datasets:
    print(f"Investigating first {N} random instances of {d} ({nrows} instances in total)")
    mismatches.append(f"{d}:")
    approx_matches.append(f"{d}:")
    correct = 0
    total = 0
    mismatch_count = 0
    approx_match_count = 0
    exact_match_count = 0

    random_permutation = np.random.permutation(nrows)[:N]

    ds = pd.DataFrame(data[d])
    for i, index in enumerate(random_permutation):
        if i % 50 == 0:
            print('Iteration:', i)
        title, text, ground_truth, _, _, _, _, _ = get_info(ds.iloc[index])
        result = ner_pipeline(text)
        merged_result = merge_result(result, model_name)
        predicted = combine_entities(merged_result)
        
        gold = [(gt['name'], gt['type']) for gt in ground_truth]
        preds = [(p['word'], p['entity']) for p in predicted]
        
        for g in gold:
            if g[1] in ['LOC', 'PER', 'ORG', 'MISC']:
                total += 1
                if g in preds: 
                    exact_match_count += 1
                elif has_approximate_match(g, preds):
                    approx_match_count += 1
                    approx_matches.append(f"{g}: {has_approximate_match(g, preds)}")
                else:
                    mismatch_count += 1
                    mismatches.append('gold: ' + str(g))
                    mismatches.append('preds:')
                    for p in preds:
                        mismatches.append(str(p))
                    
    correct = exact_match_count + approx_match_count
    accuracy = correct / total
    
    results = {}
    results['accuracy'] = accuracy

    results['exact_matches'] = {}
    results['exact_matches']['count'] = exact_match_count
    results['exact_matches']['percent'] = exact_match_count / total

    results['approx_matches'] = {}
    results['approx_matches']['count'] = approx_match_count
    results['approx_matches']['percent'] = approx_match_count / total

    results['mismatches'] = {}
    results['mismatches']['count'] = mismatch_count
    results['mismatches']['percent'] = mismatch_count / total

    results_all[d] = results
    correct_all += correct
    total_all += total
    exact_match_count_all += exact_match_count
    approx_match_count_all += approx_match_count
    mismatch_count_all += mismatch_count

accuracy_all = correct_all / total_all

print('Calculating totals...')

results = {}
results['accuracy'] = accuracy_all

results['exact_matches'] = {}
results['exact_matches']['count'] = exact_match_count_all
results['exact_matches']['percent'] = exact_match_count_all / total_all

results['approx_matches'] = {}
results['approx_matches']['count'] = approx_match_count_all
results['approx_matches']['percent'] = approx_match_count_all / total_all

results['mismatches'] = {}
results['mismatches']['count'] = mismatch_count_all
results['mismatches']['percent'] = mismatch_count_all / total_all
results_all['total'] = results

print('Done. Writing files...')

with open('results/240715_docred_mismatches.txt', 'w') as file:
    for mismatch in mismatches:
        file.write(str(mismatch) + '\n')

with open('results/240715_docred_approximate_matches.txt', 'w') as file:
    for approx_match in approx_matches:
        file.write(str(approx_match) + '\n')

with open('results/240714_docred.json', 'w') as f:
    json.dump(results_all, f, indent=4)
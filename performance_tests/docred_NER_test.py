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

def get_approx_match(gold, preds, threshold=0.8):
    gold_name, gold_type = gold
    for pred_name, pred_type in preds:
        ratio = SequenceMatcher(None, gold_name, pred_name).ratio()
        if ratio >= threshold:
            return (pred_name, pred_type)
    return None

def get_results_table(tp_exact, fp_exact, fn_exact, tp_approx, fp_approx, fn_approx):
    total_entities = tp_exact + fn_exact + tp_approx + fn_approx
    total_correct = tp_exact + tp_approx

    # Calculate metrics for exact matches
    precision_exact = tp_exact / (tp_exact + fp_exact)
    recall_exact = tp_exact / (tp_exact + fn_exact)
    f1_score_exact = 2 * (precision_exact * recall_exact) / (precision_exact + recall_exact)
    accuracy_exact = tp_exact / (tp_exact + fn_exact)

    # Calculate metrics for approximate matches (includes exact matches)
    precision_approx = (tp_exact + tp_approx) / (tp_exact + tp_approx + fp_approx)
    recall_approx = (tp_exact + tp_approx) / (tp_exact + tp_approx + fn_approx)
    f1_score_approx = 2 * (precision_approx * recall_approx) / (precision_approx + recall_approx)
    accuracy_approx = (tp_exact + tp_approx) / (tp_exact + tp_approx + fn_approx)
    results = {
        'exact_match': {
            'count': tp_exact,
            'percentage': tp_exact / total_entities,
            'accuracy': accuracy_exact,
            'precision': precision_exact,
            'recall': recall_exact,
            'f1_score': f1_score_exact
        },
        'approximate_match': {
            'count': tp_approx,
            'percentage': tp_approx / total_entities,
            'accuracy': accuracy_approx,
            'precision': precision_approx,
            'recall': recall_approx,
            'f1_score': f1_score_approx
        },
        'non-match': {
            'count': fn_approx,
            'percentage': fn_approx / total_entities
        }
    }
    return results

data = load_dataset('docred', trust_remote_code=True)
datasets = [
    ('train_annotated', 3035),
    ('train_distant', 101873)
]

model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_pipeline = pipeline('ner', model=model_name, tokenizer=tokenizer)

N = 10
mismatches = []
approx_matches = []
results_all = {}
results_all['number_of_instances'] = N

tp_exact_all, fp_exact_all, fn_exact_all = 0, 0, 0
tp_approx_all, fp_approx_all, fn_approx_all = 0, 0, 0

for d, nrows in datasets:
    print(f"Investigating first {N} random instances of {d} ({nrows} instances in total)")
    mismatches.append(f"{d}:")
    approx_matches.append(f"{d}:")

    tp_exact, fp_exact, fn_exact = 0, 0, 0
    tp_approx, fp_approx, fn_approx = 0, 0, 0

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
        preds = [(p['word'], p['entity']) for p in predicted if p['entity'] in ['LOC', 'PER', 'ORG', 'MISC']]
        
        for g in gold:
            if g in preds:
                tp_exact += 1
                tp_approx += 1
            elif get_approx_match(g, preds):
                tp_approx += 1
                fn_exact += 1
                approx_matches.append(f"{g}: {get_approx_match(g, preds)}")
            else:
                fn_exact += 1
                mismatches.append('gold: ' + str(g))
                mismatches.append('preds:')
                for p in preds:
                    mismatches.append(str(p))
    
        fp_exact += len([p for p in preds if p not in gold])
        fp_approx += len([p for p in preds if p not in gold and get_approx_match(p, gold)])
        fn_approx += len([g for g in gold if not get_approx_match(g, preds)])

    results_all[d] = get_results_table(tp_exact, fp_exact, fn_exact, tp_approx, fp_approx, fn_approx)

    tp_exact_all += tp_exact
    fp_exact_all += fp_exact
    fn_exact_all += fn_exact
    tp_approx_all += tp_approx
    fp_approx_all += fp_approx
    fn_approx_all += fn_approx

print('Calculating totals...')

results_all['total'] = get_results_table(tp_exact_all, fp_exact_all, fn_exact_all, tp_approx_all, fp_approx_all, fn_approx_all)

print('Done. Writing files...')

with open('results/240715_docred_mismatches.txt', 'w') as file:
    for mismatch in mismatches:
        file.write(str(mismatch) + '\n')

with open('results/240715_docred_approximate_matches.txt', 'w') as file:
    for approx_match in approx_matches:
        file.write(str(approx_match) + '\n')

with open('results/240714_docred.json', 'w') as f:
    json.dump(results_all, f, indent=4)
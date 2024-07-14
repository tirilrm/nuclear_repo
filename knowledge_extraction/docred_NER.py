from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd
import itertools

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

dataset = load_dataset('docred', trust_remote_code=True)
ds = pd.DataFrame(dataset['train_annotated'])
nrows = 3053

model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_pipeline = pipeline('ner', model=model_name, tokenizer=tokenizer)

correct = 0
total = 0
for i in range(nrows):
    if i % 50 == 0:
        print('Iteration:', i)
    title, text, ground_truth, _, _, _, _, _ = get_info(ds.iloc[i])
    result = ner_pipeline(text)
    merged_result = merge_result(result, model_name)
    predicted = combine_entities(merged_result)

    gold = [(gt['name'], gt['type']) for gt in ground_truth]
    preds = [(p['word'], p['entity']) for p in predicted]
    
    for e in gold:
        if e[1] in ['LOC', 'PER', 'ORG', 'MISC']:
            total += 1
            if e in preds: # important: based on EXACT match (should probably use a similarity measure instead)
                correct += 1

accuracy = correct / total
print(f'Accuracy: {accuracy:.2f}') #0.71
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
import pandas as pd
from relation_extraction.old._RE import join_text
import json
import importlib
import time

import prepare_articles_data
importlib.reload(prepare_articles_data)
from prepare_articles_data import identify_entities, create_entity_pairs, get_keywords, move_to_root

start = time.time()

print('Loading model...')
model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)

print('Loading dataset...')
dataset = load_dataset('docred', trust_remote_code=True)

ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

def make_paragraphs(sents):
    paragraphs = []
    for word_list in sents:
        paragraph = ' '.join(word_list)
        paragraphs.append(paragraph)
    return paragraphs

datasets = [
    'validation',
    'test',
    'train_annotated',
    #'train_distant'
]

custom_keywords = get_keywords()
for datatype in datasets:
    ds = pd.DataFrame(dataset[datatype])
    context_and_pairs = []
    length = len(ds)

    for i in range(length):

        if i % 9 == 0:
            print(f"{datatype}: {(i/length)*100:.3f}%")

        elems = {}
        sents = ds.iloc[i]['sents']
        paragraphs = make_paragraphs(sents)
        entities = identify_entities(paragraphs, custom_keywords)
        pairs = create_entity_pairs(entities)

        elems['datatype'] = datatype
        elems['id'] = i
        elems['context'] = join_text(sents, fancy=False)
        elems['pairs'] = pairs
        context_and_pairs.append(elems)

    print(f'Saving {datatype} data')
    move_to_root()
    with open('ignore/docred_' + datatype +'_context_and_pairs.json', 'w', encoding='utf-8') as file:
        json.dump(context_and_pairs, file, indent=4, ensure_ascii=False)

end = time.time()

print(f"Finished preparing datasets in {(end-start)/60:.2f} minutes")
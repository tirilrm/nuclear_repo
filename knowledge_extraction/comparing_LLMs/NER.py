from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import time

#kaggle = load_dataset('rjac/kaggle-entity-annotated-corpus-ner-dataset')
wikiann = load_dataset('wikiann', 'en')

models = {
    'BERT': 'dslim/bert-base-NER',
    'DistilBERT': 'dslim/distilbert-NER',
    'RoBERTa': 'Jean-Baptiste/roberta-large-ner-english'
}


example = wikiann['test']['tokens'][0]

def join_tokens(tokens):
    string = ''
    for token in tokens:
        if token.isalnum():
            string = string + ' ' + token
        else:
            string = string + token
    string = string.strip()
    return string

def test_model(model_name, wiki_dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipeline = pipeline('ner', model=model_name, tokenizer=tokenizer)

    all_preds = []
    true_preds = []
    
    print('Currently testing:', model_name)
    start_time = time.time()

    tokens = wiki_dataset['tokens']
    tags = wiki_dataset['ner_tags']
    spans = wiki_dataset['spans']

    for i in range(len(tokens)):
        text = join_tokens(tokens[i])
        result = ner_pipeline(text)
        print(result)

    end_time = time.time()
    duration = end_time - start_time
    print('Time:', duration)


test_model(models['BERT'], wikiann[:10])

'''for model in models:

    model_name = models[model]
    qa_pipeline = pipeline('ner', model=model_name)'''

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

def test_model(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipeline = pipeline('ner', model=model_name, tokenizer=tokenizer)

    all_preds = []
    true_preds = []
    
    print('Currently testing:', model_name)
    start_time = time.time()
    
    for example in dataset['test']:
        tokens = example[tokens]
        text = join_tokens(tokens)
        result = ner_pipeline(text)
        


'''for model in models:

    model_name = models[model]
    qa_pipeline = pipeline('ner', model=model_name)'''

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
                if entity['word'].startswith('##') and entity['start'] == current['end']:
                    current['word'] += entity['word'][2:]
                    current['end'] = entity['end']
                    current['score'] = min(current['score'], entity['score'])
                elif entity['start'] == current['end'] and entity['entity'][2:] == current['entity'][2:]:
                    current['word'] += entity['word']
                    current['end'] = entity['end']
                    current['score'] = min(current['score'], entity['score'])
                elif entity['start'] + 1 == current['end'] and entity['entity'][2:] == current['entity'][2:]:
                    current['word'] += ' ' + entity['word']
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

def combine_entities(entities):
    combined_entities = []
    i = 0
    while i < len(entities):
        current_entity = entities[i]
        if current_entity['entity'].startswith('B-'):
            entity_type = current_entity['entity'][2:]
            combined_entity = {
                'entity': entity_type,
                'score': current_entity['score'],
                'start': current_entity['start'],
                'end': current_entity['end'],
                'word': current_entity['word']
            }
            j = i + 1
            while j < len(entities) and entities[j]['entity'] == f'I-{entity_type}' and entities[j]['start'] == combined_entity['end'] + 1:
                combined_entity['word'] += ' ' + entities[j]['word']
                combined_entity['end'] = entities[j]['end']
                combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                j += 1
            combined_entities.append(combined_entity)
            i = j
        else:
            i += 1
    return combined_entities
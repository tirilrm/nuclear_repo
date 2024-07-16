# The functions combine_entities and merge_result are from _NER
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
            while j < len(entities):
                if entities[j]['word'] == '-':
                    combined_entity['word'] += entities[j]['word']
                    combined_entity['end'] = entities[j]['end']
                    combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                    j += 1
                elif entities[j]['entity'] == f'I-{entity_type}' and (entities[j]['start'] == combined_entity['end'] + 1):
                    combined_entity['word'] += ' ' + entities[j]['word']
                    combined_entity['end'] = entities[j]['end']
                    combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                    j += 1
                elif (entities[j-1]['word'] == '-'):
                    combined_entity['word'] += entities[j]['word']
                    combined_entity['end'] = entities[j]['end']
                    combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                    j += 1
                else:
                    break
            combined_entities.append(combined_entity)
            i = j
        else:
            i += 1
    return combined_entities

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

def join_text(sents, fancy=True):
    '''
    Make a paragraph out of given sentences and words, 
    with special rules for specific chars
    '''
    text = ''
    for sent in sents:
        sentence = ''
        for i, word in enumerate(sent):
            if fancy:
                if i == 0:
                    sentence += word
                else:
                    prev_word = sent[i-1]
                    if word.isalnum() and prev_word.isalnum():
                        sentence += ' ' + word
                    elif prev_word in '(-':
                        sentence += word
                    elif word in '"\'' and prev_word.isalnum():
                        sentence += ' ' + word
                    elif word in ')-':
                        sentence += word
                    elif word == '(':
                        sentence += ' ' + word
                    elif not word.isalnum() and prev_word[:-1].isalnum():
                        sentence += word
                    elif word[:-1].isalnum() and not prev_word.isalnum():
                        sentence += ' ' + word
                    else:
                        sentence += word
            else:
                sentence += ' ' + word
        text += sentence.strip() + '\n'
    
    return text.strip()
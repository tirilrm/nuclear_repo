from transformers import pipeline, AutoTokenizer
import re
from relik import Relik
from relik.inference.data.objects import RelikOutput
from dataclasses import asdict

class KnowledgeExtractor():
    def __init__(self, model_name, custom_keywords):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_pipeline = pipeline('ner', model=model_name, tokenizer=self.tokenizer)
        self.custom_keywords = custom_keywords
        self.relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")
    
    ############################
    #### Relation Extractor ####
    ############################
    def extract_relik_triplets(self, sents):
        extracted_triplets = []

        relik_out: RelikOutput = self.relik(sents)

        for output in relik_out:
            output_dict = asdict(output)
            triplets = output_dict['triplets']
            if triplets:
                for triplet in triplets:
                    head = triplet.subject.text
                    tail = triplet.object.text
                    label = triplet.label
                    confidence = triplet.confidence
                    extracted_triplets.append({
                        'head': head,
                        'relation': label,
                        'tail': tail,
                        'confidence': confidence
                    }) 
        
        return extracted_triplets


    ##########################
    #### Entity Extractor #### (Re-used same code as in CustomDataset class for RE training)
    ##########################

    def merge_result(self, entities):
        '''
        Merges consecutive entities output from the NER model into single entities.

        This function combines consecutive sub-word tokens or consecutive words 
        that belong to the same named entity type into a single entity. The function 
        works specifically with models 'dslim/bert-base-NER' and 'dslim/distilbert-NER'.

        Args:
            entities (list): List of entity dictionaries. Each dictionary contains 
                            'word', 'start', 'end', 'entity', and 'score' keys.

        Returns:
            list: List of merged entity dictionaries.
        '''
        merged_entities = []
        current = None

        if self.model_name not in ['dslim/bert-base-NER', 'dslim/distilbert-NER']:
            raise ValueError('NER model not compatible.')

        for entity in entities:
            if current is None:
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
        
        return merged_entities

    def combine_entities(self, entities):
        '''
        Combines entities that are split across multiple tokens, handling cases where entities span multiple tokens 
        or are separated by hyphens.

        Args:
            entities (list): List of entity dictionaries, where each dictionary contains 'entity', 'score', 'start', 'end', and 'word'.

        Returns:
            list: List of combined entity dictionaries, each containing a merged entity with its type, score, start, end, and word.

        The function processes entities by iterating through them and merging tokens that are part of the same entity.
        It considers entities that start with 'B-' (beginning) and merges subsequent tokens marked with 'I-' (inside).
        It also handles hyphens by merging tokens before and after the hyphen as part of the same entity.
        The combined entity retains the minimum score, earliest start position, and latest end position from the merged tokens.
        '''
        
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
                    elif entities[j-1]['word'] == '-':
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
    
    def extract_entities(self, sents):
        '''
        Extracts and combines entities from a list of sentences using 
         1) NER pipeline and merging functions;
         2) based on keyword watch if keywords given

        Args:
            sents (list): List of sentences (strings) to process for named entity recognition.

        Returns:
            list: List of combined entity dictionaries, each containing an entity type, score, start, end, and word.
        '''
        all_entities = []

        for i, sent in enumerate(sents):
            entities = []

            # NER entities
            ner_result = self.ner_pipeline(sent)
            merged_result = self.merge_result(ner_result)
            joined_result = self.combine_entities(merged_result)

            for result in joined_result:
                word = result['word']
                if '##' not in word and 'of' not in word:
                    entity = {
                        #'s_id': i,
                        'word': result['word'],
                        'entity': result['entity'],
                        #'start': result['start'],
                        #'end': result['end']
                    }
                    entities.append(entity)
        
            # Keyword entities
            if self.custom_keywords:
                for label, terms in self.custom_keywords.items():
                    # Precompile the regex patterns
                    term_patterns = [re.compile(re.escape(term)) for term in terms]
                    for pattern in term_patterns:
                        for match in pattern.finditer(sent):
                            entity = {
                                #'s_id': i,
                                'word': match.group(),
                                'entity': label,
                                #'start': match.start(),
                                #'end': match.end()
                            }
                            entities.append(entity)
            all_entities.extend(entities)
        
        return all_entities
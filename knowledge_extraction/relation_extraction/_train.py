from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from torch import nn
import torch
import re
from _RE import merge_result, combine_entities

model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

custom_keywords = {
    'nuclear power plant': 'SPECIFIC_TERM',
    # Add more custom terms and labels here
}

paragraphs = [
    "Canada is planning to build a new nuclear power plant. This decision comes as part of a broader energy strategy. In other news, Barack Obama visited Microsoft in Seattle, highlighting the importance of technology in modern economies.",
]

def identify_entities(paragraphs, custom_keywords):
    entities = []
    for paragraph in paragraphs:
        raw = ner_pipeline(paragraph)
        raw_merged = merge_result(raw, model_name)
        ner_results = combine_entities(raw_merged)
        paragraph_entities = []
        
        # Add standard NER results
        for result in ner_results:
            entity = {
                'word': result['word'],
                'entity': result['entity'],
                'start': result['start'],
                'end': result['end']
            }
            paragraph_entities.append(entity)
        
        # Add custom keyword results
        for term, label in custom_keywords.items():
            for match in re.finditer(term, paragraph):
                entity = {
                    'word': term,
                    'entity': label,
                    'start': match.start(),
                    'end': match.end()
                }
                paragraph_entities.append(entity)
        
        entities.append(paragraph_entities)
    return entities

identified_entities = identify_entities(paragraphs, custom_keywords)
print(identified_entities)

def create_entity_pairs(entities):
    '''
    Returns a list of entity pairs where 1st and 2nd aren't the same entity
    '''
    pairs = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:
                pairs.append((entity1, entity2))
    return pairs

def create_context_pairs(paragraph, entity_pairs):
    contexts = []
    for e1, e2 in entity_pairs:
        context = paragraph
        contexts.append({
            'context': context,
            'entity1': e1['word'],
            'entity2': e2['word'],
            'entity1_pos': e1['start'],
            'entity2_pos': e2['start']
        })
    return contexts

context_data = []
for paragraph, entities in zip(paragraphs, identified_entities):
    entity_pairs = create_entity_pairs(entities)
    contexts = create_context_pairs(paragraph, entity_pairs)
    context_data.extend(contexts)

print(context_data)

class RelationExtractionModel(nn.Module):
    def __init__(self, distilbert_model_name='distilbert-base-uncased', lstm_hidden_size=128, num_classes=97):
        super(RelationExtractionModel, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(distilbert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, _ = self.lstm(bert_outputs)
        logits = self.classifier(lstm_output[:, 0, :])
        return logits
    
relation_model = RelationExtractionModel()

def prepare_relation_extraction_data(context_data, tokenizer):
    inputs = []
    for data in context_data:
        input = tokenizer(data['context'], return_tensors='pt')
        inputs.append(input)
    return inputs

relation_inputs = prepare_relation_extraction_data(context_data, tokenizer)

# Example labels for training (dummy labels)
relation_labels = torch.tensor([0] * len(relation_inputs))

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(relation_model.parameters(), lr=2e-5)

# Training loop (simplified)
epochs = 3
for epoch in range(epochs):
    relation_model.train()
    total_loss = 0
    for input, label in zip(relation_inputs, relation_labels):
        input_ids = input['input_ids'].to('cuda')
        attention_mask = input['attention_mask'].to('cuda')
        label = label.to('cuda')

        optimizer.zero_grad()
        outputs = relation_model(input_ids, attention_mask)
        loss = criterion(outputs, label.unsqueeze(0).float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(relation_inputs)}")

from transformers import AutoModelForTokenClassification
import torch.nn as nn



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

#relation_inputs = prepare_relation_extraction_data(context_data, tokenizer)

# Example labels for training (dummy labels)
#relation_labels = torch.tensor([0] * len(relation_inputs))

# Training setup
#criterion = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.Adam(relation_model.parameters(), lr=2e-5)

# Training loop (simplified)
def train(relation_inputs, relation_labels, relation_model, criterion, optimizer, epochs = 100):
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


'''from transformers import AutoTokenizer, DistilBertModel, pipeline, DistilBertTokenizer, DistilBertForTokenClassification
import torch
import torch.nn as nn

class RelationExtractionModel(nn.Module):

    def __init__(self, distilbert_model_name='dslim/distilbert-NER', lstm_hidden_size=128, num_classes=97):
        super(RelationExtractionModel, self).__init__()
        self.bert = DistilBertForTokenClassification.from_pretrained(distilbert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, _ = self.lstm(bert_outputs)
        logits = self.classifier(lstm_output[:, 0, :])
        return logits
    
model = RelationExtractionModel()
criterion = nn.BCEWithLogitsLoss()  # or another appropriate loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
'''
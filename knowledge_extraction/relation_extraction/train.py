from transformers import AutoTokenizer, DistilBertModel, pipeline, DistilBertTokenizer, DistilBertForTokenClassification
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

import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time

import importlib
import CustomDataset
importlib.reload(CustomDataset)
import RE_BiLSTM as RE_BiLSTM
importlib.reload(RE_BiLSTM)
import CustomLoss
importlib.reload(CustomLoss)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'dslim/distilbert-NER'

try:
    with open('keyword_matching/directory.pkl', 'rb') as file:
        keywords = pickle.load(file)
except FileNotFoundError or FileExistsError:
    with open('directory.pkl', 'rb') as file:
        keywords = pickle.load(file)

input_size = 512
output_size = 768
num_layers = 4          # may require tuning
hidden_size = 256       # may require tuning
num_classes = 97        # 96 different relations plus '0' for no relation
learning_rate = 0.001   # may require tuning
batch_size = 32
num_epochs = 5
PAIR_EMBEDDING_WIDTH = 1540
PAIR_EMBEDDING_LENGTH = 3000

# Prepare datasets
length = 100
train = CustomDataset.CustomDocREDDataset(
    dataset='train_annotated',
    input_size=input_size,
    model_name=model_name,
    custom_keywords=keywords,
    device=device,
    length = length*2
)
test = CustomDataset.CustomDocREDDataset(
    dataset='test',
    input_size=input_size,
    model_name=model_name,
    custom_keywords=keywords,
    device=device,
    length = length
)
val = CustomDataset.CustomDocREDDataset(
    dataset='validation',
    input_size=input_size,
    model_name=model_name,
    custom_keywords=keywords,
    device=device,
    length = length
)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=CustomDataset.custom_collate_fn)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=CustomDataset.custom_collate_fn)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=CustomDataset.custom_collate_fn)

# Load model
model = RE_BiLSTM.RelationExtractorBRNN(
    input_size=output_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    pair_embedding_width=PAIR_EMBEDDING_WIDTH,
    pair_embedding_length=PAIR_EMBEDDING_LENGTH,
    model_name=model_name,
    device=device
).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = CustomLoss.CustomLoss(threshold=0.8)

start = time.time()
for epoch in range(num_epochs):
    current = time.time()
    time_passed = current - start
    if time_passed >= 3*24*60*60:
        print('Exceeded maximum amount of time. Saving models...')
        break

    model.train()
    total_loss = 0

    for batch in train_loader:
        text_embeddings = batch['text_embeddings'].to(device)
        pair_embeddings = batch['pair_embeddings'].to(device)
        triplet_embeddings = batch['triplet_embeddings'].to(device)

        # Forward pass
        preds = model(text_embeddings, pair_embeddings) # shape [batch_size, num_pairs]

        # Calculate loss
        loss = loss_fn(pair_embeddings, preds, triplet_embeddings)

        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            text_embeddings = batch['text_embeddings'].to(device)
            pair_embeddings = batch['pair_embeddings'].to(device)
            triplet_embeddings = batch['triplet_embeddings'].to(device)

            preds = model(text_embeddings, pair_embeddings)

            val_loss = loss_fn(pair_embeddings, preds, triplet_embeddings)
            total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    
model_save_path = "model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

checkpoint_path = "checkpoint.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")
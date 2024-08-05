import torch
import pickle
from torch.utils.data import DataLoader
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

input_size = 512
output_size = 768
num_layers = 4          # may require tuning
hidden_size = 256       # may require tuning
num_classes = 97        # 96 different relations plus '0' for no relation
learning_rate = 0.001   # may require tuning
threshold = 0.85
batch_size = 32
num_epochs = 5

PAIR_EMBEDDING_WIDTH = 1540
PAIR_EMBEDDING_LENGTH = 3000

keywords = {
    'FUEL': 
    [
            'u235', 'u238', 'uranium-235', 'uranium-238'
            'uranium compound', 'uranium oxide', 'uranium dioxide' 'uranium fuel',
            'nuclear fuel', 'mox', 'mox fuel', 'mixed oxide fuel',
            'plutonium', 'pu239', 'plutonium-239', 'thorium', 'actinides',
            'light water', 'heavy water'
    ],
    'FUEL_CYCLE': 
    [
        'uranium oxide', 'uranium hexafluoride', 'hex' ,'wet process', 'dry process',
        'uranium enrichment', 'gas centrifuge',
        'fuel rods', 'fuel assembly', 'low enriched fuel', 'leu', 'highly enriched fuel', 'heu',
        'high assay low enriched uranium', 'haleu', 'triso',
        'spent fuel', 'spent nuclear fuel', 'nuclear waste', 'radioactive waste',
        'spent oxide fuel', 'spent reactor fuel', 'spent fuel pools', 'spent fuel ponds',
    ],
    'SMR_DESIGN': 
    [
        'water-cooled', 'water cooled',
        'light water reactor', 'lwr',
        'heavy water reactor', 'hwr',
        'boiling water reactor', 'pressurized water reactor', 'pwr',
        'high temperature gas reactor', 'htgr', 
        'gas reactor', 'gas-cooled', 'gas cooled', 'pebble bed reactor', 'pbmr'
        'liquid metal cooled', 'liquid-metal-cooled', 'liquid metal-cooled',
        'lead-bismuth', 'sodium cooled', 'sodium-cooled'
        'molten salt reactor', 'molten salt', 'msr'
        'microreactor', 'micro reactor'
        'micro modular reactor', 'micro nuclear reactor'
    ],
    'REACTOR': 
    [
        'nuclear reactor', 'nuclear power plant', 'nuclear power reactor',
        'fast reactor',
    ],
    'SMR': 
    [
        'smr', 'small modular reactor', 
        'small nuclear reactor',
    ],
    'POLITICAL': 
    [
        'safety', 'security', 'nuclear regulation', 
        'proliferation', 'safeguards',
    ]
}

# 1. Load datasets
length = 10
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

# 2. Make data loaders for efficient batching
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=CustomDataset.custom_collate_fn)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=CustomDataset.custom_collate_fn)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=CustomDataset.custom_collate_fn)

# 3. Load the model
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

# 4. Load optimizer and custom loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = CustomLoss.CustomLoss(threshold=threshold)

# 5. Start training
start = time.time()
avg_loss = 0.0
avg_val_loss = 0.0

for epoch in range(num_epochs):
    current = time.time()
    time_passed = current-start
    if time_passed - start >= 3*24*60*60:
        print('Exceeded maximum amount of time. Saving models...')
        break

    model.train()
    total_loss = 0

    for batch in train_loader:
        text_embeddings = batch['text_embeddings']
        pair_embeddings = batch['pair_embeddings']
        triplet_embeddings = batch['triplet_embeddings']

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
            triplet_embeddings = batch['triplet_embeddings']

            preds = model(text_embeddings, pair_embeddings)

            val_loss = loss_fn(pair_embeddings, preds, triplet_embeddings)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save model and checkpoint at the end of each eopoch
    print('Saving interim model and checkpoint')
    model_save_path = f"models/model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)

    checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'val_loss': avg_val_loss
    }, checkpoint_path)

# 6. Save final model and checkpoint
model_save_path = "models/model_epoch_final.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

checkpoint_path = "checkpoints/checkpoint_final.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': avg_loss, # Using last calculated
    'val_loss': avg_val_loss # Using last calculated
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")
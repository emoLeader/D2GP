# Key points explanation:
# Reduce preprocessing data storage: Do not tokenize and store all data in advance, but tokenize on-the-fly within each batch.
# Minimize debug logging: Only print debug information at critical steps.
# Use mixed-precision training: Employ GradScaler and autocast to enable mixed-precision training, reducing VRAM usage and speeding up training.
# Free unneeded cache: Call torch.cuda.empty_cache() at the end of each epoch to release unnecessary GPU memory.

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import random
import json
import os
from torch.cuda.amp import GradScaler, autocast
import ipdb
from tqdm import tqdm

# Load demand-task data
with open('/D2GP/json/demand_tasks.json', 'r') as f:
    data = json.load(f)

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom model class using BERT to extract features
class DemandTaskModel(nn.Module):
    def __init__(self, bert_model):
        super(DemandTaskModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use the [CLS] token output as the sentence embedding
        return outputs.last_hidden_state[:, 0, :] 

# Define contrastive loss function
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor_embeddings, task_embeddings, index):
        device = anchor_embeddings.device
        B, D = anchor_embeddings.size()
        M = task_embeddings.size(0)

        # Create a mask indicating which task samples are positives for each anchor:
        # pos_mask[i, j] = 1.0 if task_embeddings[j] is a positive for anchor_embeddings[i], else 0.0
        index_tensor = torch.tensor(index, device=device)          # [M]
        anchor_ids = torch.arange(B, device=device)                # [B]

        # Expand to [B, M] so we can compare index[j] to each anchor i
        anchor_ids_expanded = anchor_ids.unsqueeze(1).expand(B, M)  # [B, M]
        index_expanded = index_tensor.unsqueeze(0).expand(B, M)      # [B, M]
        pos_mask = (anchor_ids_expanded == index_expanded).float()   # [B, M]

        # Compute similarity logits: [B, M] = (anchor_embeddings @ task_embeddings.T) / temperature
        sim_matrix = torch.matmul(anchor_embeddings, task_embeddings.t()) / self.temperature  # [B, M]
        exp_sim_matrix = torch.exp(sim_matrix)  # exponentiate

        # Denominator for each anchor i: sum over all task samples j
        denom = exp_sim_matrix.sum(dim=1, keepdim=True)  # [B, 1]

        # Numerator: sum over only positive positions per anchor i
        numerator = (pos_mask * exp_sim_matrix).sum(dim=1, keepdim=True)  # [B, 1]

        # Avoid numerical issues
        eps = 1e-8
        numerator = numerator.clamp(min=eps)
        denom = denom.clamp(min=eps)

        # InfoNCE loss for each anchor: -log(numerator_i / denom_i), then average over B
        loss = -torch.log(numerator / denom)  # [B, 1]
        return loss.mean()
    
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)
    
#     def forward(self, anchor, task, index):
#         index = torch.tensor(index)
#         total_neg_loss = 0
#         total_pos_loss = 0
#         for i in range(anchor.size(0)):
#              # Current demand index
#             demand = anchor[i].view(1, -1)
#             # Identify negative samples (tasks that do not belong to this demand)
#             negative_idx = torch.where(index != i)[0]
#             negative = task[negative_idx]
#             # Identify positive samples (tasks that belong to this demand)
#             positive_idx = torch.where(index == i)[0]
#             positive = task[positive_idx]  

#             # Negative samples label is -1
#             neg_labels = -torch.ones(negative.size(0)).to(anchor.device)
#             neg_loss = self.cosine_loss(demand, negative, neg_labels)
#             # Positive samples label is +1
#             pos_labels = torch.ones(positive.size(0)).to(anchor.device)
#             pos_loss = self.cosine_loss(demand, positive, pos_labels)
#             total_neg_loss += neg_loss
#             total_pos_loss += pos_loss

#         total_loss = (total_pos_loss + total_neg_loss) / 2
#         return total_loss

# Instantiate model and move to GPU
model = DemandTaskModel(bert_model).to(device)
loss_fn = InfoNCELoss(temperature=0.07).to(device)
# loss_fn = ContrastiveLoss(margin=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = GradScaler()

# Create a data generator function that yields batches
def create_batches(data, batch_size=8, num_neg_samples=1):
    demands = list(data.keys())
    all_tasks = {obj for objs in data.values() for obj in objs}  # Avoid repeated computation
    
    for start_idx in range(0, len(demands), batch_size):
        batch_demands = demands[start_idx:start_idx + batch_size]
        anchor_texts, positive_texts, index = [], [], []
        
        for num,demand in enumerate(batch_demands):
            pos_tasks = data[demand]
            for pos_obj in pos_tasks:
                # Add positive samples
                positive_texts.append(pos_obj)
                index.append(num)
        
        yield (batch_demands, positive_texts, index)

# Function to save the model, ensuring the directory exists
def save_model(model, epoch, save_dir='saved_models', path_prefix='demand_task_model'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{path_prefix}_epoch{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

# Training loop
num_epochs = 1000  # Adjust based on dataset size
batch_size = 16    # Adjust based on available GPU memory
save_interval = 100  # Save the model every N epochs
save_directory = "saved_models"  # Directory to store model checkpoints

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    
    # Shuffle data at the start of each epoch
    shuffled_data = list(data.items())
    random.shuffle(shuffled_data)
    shuffled_data = dict(shuffled_data)
    
    # Generate batches and store them in a list
    batches = list(create_batches(shuffled_data, batch_size=batch_size))
    print(f"Epoch {epoch + 1}/{num_epochs}, Number of batches: {len(batches)}")
    
    # Iterate over batches
    pbar = tqdm(total=len(batches))
    for i, (anchor_texts, positive_texts, index) in enumerate(batches):
        # Tokenize on-the-fly
        anchor_inputs = tokenizer(anchor_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        positive_inputs = tokenizer(positive_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        anchor_embeddings = model(**anchor_inputs).to(device)
        positive_embeddings = model(**positive_inputs).to(device)

        # Compute loss
        loss = loss_fn(anchor_embeddings, positive_embeddings, index)
        
        # Optimize with mixed-precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.update(1)
    pbar.close()
    avg_loss = total_loss / len(batches)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save model checkpoint at specified intervals
    if (epoch + 1) % save_interval == 0:
        save_model(model, epoch + 1, save_dir=save_directory)
    
    # Release unused GPU cache
    torch.cuda.empty_cache()

# Final model save
torch.save(model.state_dict(), 'demand_task_model_plus1000.pth')
print("Model saved to 'demand_task_model_plus1000.pth'")
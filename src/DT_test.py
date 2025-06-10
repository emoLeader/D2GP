import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import BertTokenizer, BertModel

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the DemandTaskModel class
class DemandTaskModel(nn.Module):
    def __init__(self, bert_model):
        super(DemandTaskModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.last_hidden_state[:, 0, :]  # Use the CLS token output as the sentence embedding

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create and load a DemandTaskModel instance
model = DemandTaskModel(bert_model).to(device)
# model.load_state_dict(torch.load('/D2GP/src/saved_models1/demand_task_model_epoch100.pth', map_location=device))
state_dict = torch.load('/D2GP/src/saved_models/demand_task_model_epoch100.pth', map_location=device)
model.load_state_dict(state_dict, strict=False)  # strict=False will ignore extra position_ids
model.eval()

# Pre-compute and store embeddings for demands
demand_embeddings = {}
# Load demand-to-task data
with open('/D2GP/json/demand_tasks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for demand in data.keys():
    with torch.no_grad():
        demand_encoding = tokenizer(demand, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        demand_embedding = model(**demand_encoding)
        demand_embeddings[demand] = demand_embedding

# Build a mapping from each task to its demand
task_to_demand = {}
for demand, tasks in data.items():
    for task in tasks:
        task_to_demand[task] = demand  # Assume each task corresponds to exactly one demand

# Load the grounded tasks dataset
with open('/D2GP/json/grounded_tasks.json', 'r', encoding='utf-8') as f:
    grounded_tasks = json.load(f)

# Define a function that returns the top K objects most relevant to a demand, along with similarity scores
def get_top_k_objects(model, demand, objects, top_k):
    # Set the model to evaluation mode
    model.eval()
    
    # Compute the embedding for the demand
    with torch.no_grad():
        demand_encoding = tokenizer(demand, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        demand_embedding = model(**demand_encoding).squeeze(0)  # shape: (hidden_size,)

    # Compute each object's embedding and its similarity to the demand
    object_scores = []
    with torch.no_grad():
        for obj in objects:
            obj_encoding = tokenizer(obj, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            obj_embedding = model(**obj_encoding).squeeze(0)  # shape: (hidden_size,)
            # Compute cosine similarity
            similarity = F.cosine_similarity(demand_embedding, obj_embedding, dim=0)
            object_scores.append((obj, similarity.item()))
    
    # Sort by similarity from high to low, then select the top K objects
    sorted_objects = sorted(object_scores, key=lambda x: x[1], reverse=True)
    top_k_objects = sorted_objects[:top_k]
    
    # Extract similarity scores
    scores = [score for _, score in top_k_objects]
    
    # Return objects paired with their similarity scores
    top_k_objects_with_scores = [(obj, score) for (obj, score) in top_k_objects]
    
    return top_k_objects_with_scores

def find_tasks_for_same_demand(input_task, top_k=10):
    # Check if input_task exists in the task-to-demand mapping
    matched_demand = task_to_demand.get(input_task, None)
    
    if matched_demand is not None:
        # If a corresponding demand is found, find related tasks
        print(f"Found demand for input task '{input_task}': {matched_demand}")
    else:
        # If no corresponding demand is found, treat the input task as a demand
        matched_demand = input_task
        print(f"No demand found for input task '{input_task}', treating it directly as a demand.")

    # Get similar tasks from grounded_tasks
    similar_tasks_with_scores = get_top_k_objects(model, matched_demand, grounded_tasks, top_k)
    print(f"Tasks related to '{matched_demand}' and their similarity scores: {similar_tasks_with_scores}")

    # Sort similar tasks by similarity from high to low and select the top K
    sorted_similar_tasks = sorted(similar_tasks_with_scores, key=lambda x: x[1], reverse=True)
    top_k_similar_tasks = sorted_similar_tasks[:top_k]

    return top_k_similar_tasks, matched_demand

# Function: find the top K tasks that satisfy a given demand
def find_tasks_for_input_demand(input_demand, top_k=10):
    # Compute the embedding for the input demand
    with torch.no_grad():
        demand_encoding = tokenizer(input_demand, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        input_demand_embedding = model(**demand_encoding).squeeze(0)  # shape: (hidden_size,)
    
    # Store the similarity between each grounded task and the input demand
    task_scores = []
    with torch.no_grad():
        for task in grounded_tasks:  # Iterate over tasks in the grounded_tasks file
            # Compute the embedding for the task
            task_encoding = tokenizer(task, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            task_embedding = model(**task_encoding).squeeze(0)  # shape: (hidden_size,)
            
            # Compute cosine similarity between task and input demand
            similarity = F.cosine_similarity(input_demand_embedding, task_embedding, dim=0)
            task_scores.append((task, similarity.item()))
    
    # Sort by similarity from high to low and select the top K tasks
    sorted_tasks = sorted(task_scores, key=lambda x: x[1], reverse=True)
    top_k_tasks = sorted_tasks[:top_k]
    
    return top_k_tasks

# Test: given a demand input, find tasks that satisfy the demand
def test_demand_input(input_demand, top_k=10):
    top_k_tasks = find_tasks_for_input_demand(input_demand, top_k)
    print(f"Input demand: '{input_demand}'")
    print("Tasks most relevant to the demand:")
    for task, score in top_k_tasks:
        print(f"{task} (Similarity: {score:.4f})")

    return top_k_tasks

# Test: find tasks that satisfy the same demand
def test_input(input_task, top_k=10):
    # Uncomment if you want to check existence in data
    # if input_task in data:
    similar_tasks_with_scores, matched_demand = find_tasks_for_same_demand(input_task, top_k=top_k)
    # else:
    #     print("No tasks found for that demand.")
    #     return [], None

    if matched_demand:
        print(f"Matched demand for input task '{input_task}': {matched_demand}")
        if similar_tasks_with_scores:
            print("Tasks that satisfy the same demand are:")
            for task, score in similar_tasks_with_scores:
                print(f"{task} (Similarity: {score:.4f})")
        else:
            print("No other tasks satisfy that demand.")
    else:
        print("No matching demand or tasks found.")

    return similar_tasks_with_scores, matched_demand

# Input task tests (uncomment to run)
# input_task1 = "serve juice"
# input_task1 = "toast the bread"
# similar_tasks_with_scores, matched_demand = test_input(input_task1, top_k=1000)

# input_task2 = "make a sandwich"
# similar_tasks_with_scores, matched_demand = test_input(input_task2, top_k=10)

# input_task3 = "serve snacks"
# similar_tasks_with_scores, matched_demand = test_input(input_task3, top_k=10)

# input_task4 = "wash plate"
# similar_tasks_with_scores, matched_demand = test_input(input_task4, top_k=10)

# input_task5 = "provide juice"
# similar_tasks_with_scores, matched_demand = test_input(input_task5, top_k=10)

# Demand input tests (uncomment to run)
# input_demand4 = "I need to have a meeting"
# top_tasks = test_demand_input(input_demand4, top_k=10)

# input_demand1 = "I need something to drink"
# top_tasks = test_demand_input(input_demand1, top_k=10)

# input_demand2 = "I am hungry and need something to eat"
# top_tasks = test_demand_input(input_demand2, top_k=10)

# input_demand3 = "I want to relax and sit comfortably"
# top_tasks = test_demand_input(input_demand3, top_k=10)

# input_demand5 = "I want to store and keep food fresh"
# top_tasks = test_demand_input(input_demand5, top_k=10)
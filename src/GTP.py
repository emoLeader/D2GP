# version  openai==0.28.0

import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
import numpy as np
import ast
import tools.fix as fix
import time
import recursive_traversal as RT


GPU = 0
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)

# see comments above for all options
translation_lm_id = 'stsb-roberta-large'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# maximum number of steps to be generated
MAX_STEPS = 30  #20  

# early stopping threshold based on matching score and likelihood score
CUTOFF_THRESHOLD = 0.5 # 0.5  # 0.8

# hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
P = 0.5

# weighting coefficient used to rank generated samples
BETA =0.2 # 0.2   # 0.3 (The smaller this is, the higher the score of each generated step)

openai.api_base = ""
openai.api_key = ""

sampling_params = \
                {
                "max_tokens": 10,
                "temperature": 0.6,
                "top_p": 0.9,
                "n": 10,
                "logprobs": True,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": '\n'
                }

'''
Initialize the Planning LM from OpenAI API.
The underlying API is abstracted by creating a generator function with a unified interface.
'''

def generator(prompt, sampling_params):
    response = openai.ChatCompletion.create\
    (
        # model = "gpt-3.5-turbo",
        # model="qwen-max",
        model = "gpt-4",
        messages = [{"role": "user", "content": prompt}],
        # **sampling_params
        max_tokens = 50,
        temperature = 0.6,
        top_p = 0.9,
        n = 10,
        logprobs = True,
        presence_penalty = 0.5,
        frequency_penalty = 0.3,
        stop = '\n'
    )

    # Extracting generated samples and calculating mean log probabilities
    generated_samples = []
    mean_log_probs = []

    # Use the actual number of choices returned in the response
    for i in range(len(response['choices'])):
        content = response['choices'][i]['message']['content']
        generated_samples.append(content.strip().lower())

        if 'logprobs' in response['choices'][i]:
            try:
                # Ensure 'content' key exists in 'logprobs'
                if 'content' in response['choices'][i]['logprobs']:
                    mean_log_probs.append(np.mean([token['logprob'] for token in response['choices'][i]['logprobs']['content']]))
                else:
                    mean_log_probs.append(None)
            except KeyError:
                mean_log_probs.append(None)
        else:
            mean_log_probs.append(None)

    return generated_samples, mean_log_probs

'''
General-purpose LLMs to obtain responses
'''
# General LLM 1 (lenient style)
def get_response(prompt):
    response = openai.ChatCompletion.create\
    (
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
        top_p=0.5,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    text = response['choices'][0]['message']['content']
    return text

# General LLM 2 (strict style)
def get_response_strict(prompt):
    response = openai.ChatCompletion.create\
    (
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    text = response['choices'][0]['message']['content']
    return text


'''
Initialize the Translation LM and create embeddings for:
- all available actions (for action translation)
- all available example task names (for finding related examples)
'''

# initialize Translation LM
translation_lm = SentenceTransformer(translation_lm_id).to(device)

# create action embeddings using Translated LM
with open('/D2GP/json/available_actions.json', 'r') as f:
    action_list = json.load(f)
action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# create example task embeddings using Translated LM 
with open('/D2GP/json/available_examples.json', 'r') as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory


# Save action embeddings
np.save('/D2GP/src/action_list_embedding.npy', action_list_embedding.cpu().numpy())

# Save example task embeddings
np.save('/D2GP/src/example_task_embedding.npy', example_task_embedding.cpu().numpy())


# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score

# # Load action embeddings
# action_list_embedding = np.load('/D2GP/src/action_list_embedding.npy')
# action_list_embedding = torch.tensor(action_list_embedding).to(device)

# # Load example task embeddings
# example_task_embedding = np.load('/D2GP/src/example_task_embedding.npy')
# example_task_embedding = torch.tensor(example_task_embedding).to(device)

action_list_storage = []
effect_list_storage = []

# define query task
# task = 'serve coke'
# task = 'wash plate'
# task = 'provide a cup of water'
# task = 'close the windows'
# task = 'prepare the dinnertable'
task = 'slice an apple'
# task = 'water the flowers'
# task = 'Wipe the window'
# task = 'heat the bread'
# task = 'boil water'
# task = 'wash cloths'
# task = 'turn on the TV'
# task = 'make breakfast'
# task = 'make coffee'
# task = 'serve coke to Tom'
# task = 'wash bowl'
# task = 'clean the table'
# task = 'wipe the table'
# task = 'make a sandwich' 
# task = 'water flowers'
# task = 'serve a glass of bear'
# task = 'mop the floor' 
# task = 'move the apple to the desk'
# task = 'serve a glass of juice to Tom'
# task = 'put an apple on the plate'

# find most relevant example
example_idx, _ = find_most_similar(task, example_task_embedding)
example = available_examples[example_idx]

curr_prompt = f'{example}\n\nTask: {task}\nOnly one step is generated at a time with no explanatory information\n'

# print example and query task
print('-'*10 + ' GIVEN EXAMPLE ' + '-'*10)
print(example)
print('-'*10 + ' EXAMPLE END ' + '-'*10)
print(f'\nTask: {task}')

# Determine whether the task has been completed
def check_task_completion_llm(task, action_effect):
    prompt = f'''
        You are given a task: "{task}". Firstly, analyze what this task needs to accomplish carefully. 
        Here are the steps taken to complete the task and their effects:
        {action_effect}
        please judge whether these actions can complete the task by the effect of each action after execution.
        If they do, respond with "yes"; if not, respond with "no". Do not output any explanation
        '''
    response = openai.ChatCompletion.create\
    (
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    text = response['choices'][0]['message']['content']
    return text

# Initialize previous_action and previous_action_2 for duplicate checking
previous_action = None
previous_action_2 = None

# Use threshold to decide whether to keep the action; use effect to judge whether to terminate action generation
for step in range(1, MAX_STEPS + 1):
    best_overall_score = -np.inf
    best_action = None
    
    # Query one-step action candidates
    samples, log_probs = generator(curr_prompt + f'\nStep {step}:', sampling_params)
    
    for sample, log_prob in zip(samples, log_probs):
        # Find the most similar action and calculate the matching score
        most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
        
        # Combine similarity and log probability to compute the total score, weighted by BETA
        overall_score = matching_score + BETA * log_prob
        
        translated_action = action_list[most_similar_idx]
        
        # Penalize if the current action is the same as either of the previous two steps
        if step > 1 and (translated_action == previous_action or translated_action == previous_action_2):
            overall_score -= 0.8

        # Update best action if current score is higher
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_action = translated_action
    
    # Skip the best action if its score is below the threshold
    if best_overall_score < CUTOFF_THRESHOLD:
        print(f'\n[Skipped current action: Best score below threshold ({best_overall_score} < {CUTOFF_THRESHOLD})]')
        continue
    
    # Update previous actions and format the current action
    previous_action_2 = previous_action
    previous_action = best_action
    formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ').replace('Step {step}: ', '')
    curr_prompt += f'\nStep {step}: {formatted_action}'
    
    time.sleep(10)

    # Generate and store the effect of the current action
    effect_prompt = curr_prompt + f'\nEffect of Step {step}:'
    effect = get_response(effect_prompt + f'Please answer "effect" succinctly, confined to one sentence')
    formatted_effect = effect.capitalize()
    
    action_list_storage.append(f'Step {step}: {formatted_action}')
    effect_list_storage.append(f'Step {step}: {formatted_action}, Effect: {formatted_effect}')
    
    print(f'Step {step}: {formatted_action}')
    print(f'Effect of Step {step}: {formatted_effect}')
    print(best_overall_score)
    
    # Check whether the task is completed
    true_label = check_task_completion_llm(task, effect_list_storage)
    print(true_label)
    if true_label.lower() == "yes":
        print(f'\n[Task completed at step {step}]')
        break
'''
Print the initial task plan
'''
llm_output = '\n'.join(action_list_storage)
                    
# print("\nGenerated Actions:\n" + llm_output)
print('\n' + '-'*10 + ' Initial Actions ' + '-'*10)
print(llm_output)

# Load all object names
with open('/D2GP/json/all_objects.json', 'r') as f:
    object_list = json.load(f)

#TSGP
prompt_object = f'For the task "{task}", which of the following objects are relevant: {", ".join(object_list)}? Please list all the relevant objects. No explanations needed. For example, ["a","b",...]'
relevant_objects = get_response_strict(prompt_object)

print("List of objects relevant to the task:")
print(relevant_objects)
relevant_objects = ast.literal_eval(relevant_objects)

# Check if the number of relevant objects exceeds 10
if len(relevant_objects) > 10:
    prompt_object = f'For the task "{task}", its initial list of actions is: {llm_output}, which of the following objects are relevant: {", ".join(object_list)}? Please list all the relevant objects. No explanations needed. For example, ["a","b",...]'
    
    relevant_objects = get_response_strict(prompt_object)
    
    print("Updated list of objects relevant to the task:")
    print(relevant_objects)
    
    relevant_objects = ast.literal_eval(relevant_objects)

# Based on the relevant objects, recursively traverse bottom-up to obtain a pruned scene graph (including location and state information)
with open('/D2GP/json/position_relationships_state.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

all_location_information_dic,  all_location_information_list, all_related_objects= RT.find_chain_info(data, relevant_objects)
all_location_information_list = [res.replace("isOn", "is on").replace("isIn", "is in") for res in all_location_information_list]
all_location_information_list = list(set(all_location_information_list))
all_related_objects = list(set(all_related_objects))
print(all_related_objects)
print("\n")

# Replace keywords in location relationships for natural phrasing
for key in all_location_information_dic:
    all_location_information_dic[key] = [res.replace("isOn", "is on").replace("isIn", "is in") for res in all_location_information_dic[key]]

# Print all location information
print("Location information: "+"\n")
for obj, info in all_location_information_dic.items():
    print(f"Information for {obj}:")
    for line in info:
        print(f"  - {line}")
print("\n")

state_info = RT.get_has_state(data, all_related_objects)
state_info = list(state_info)
state_info = [res.replace("hasState", "is") for res in state_info]

# Print object state information
print("State Information: ", state_info)


 
prompt_grounded = f'''
    For the service task {task}, the initial action list of solving tasks is as follows: {llm_output}.
    In a real home environment, the task-related objects are {all_related_objects}, their location information is {all_location_information_list}, and their state information is {state_info}.
    Modify the initial action list based on the task-related objects' environment information and the positions of these objects. Make sure to analyze the status of the objects based on common sense knowledge and the provided state information (e.g., is an object dirty, broken, or unavailable).
    Specifically:
    1. - If an object is broken or unusable, please prioritize replacing it with an appropriate usable object from {all_related_objects}.    
       - If no suitable replacement is available, use the original object if it is still functional enough to complete the action.
    2. If an object is already in the correct state (e.g., clean, functional), do not repeat unnecessary actions (e.g., don't "open a cupboard" if it is already opened).
    3. Ensure actions are modified to maintain logical consistency with the new STATE of related objects and LOCATIONS.
    4. If an action does not align with the current state or the object status (e.g., trying to use a broken mug), replace it with a valid alternative action.
    
    Strictly and clearly, output the modified action list in the format: "Step n: action", making sure all the changes reflect the correct handling of objects.
    ----------------------------------------
    Example of the desired output format:
    Step 1: action 1
    Step 2: action 2
    Step 3: action 3
    Step 4: action 4
    Step 5: action 5
    Step 6: ...
    ...
'''

# prompt_grounded = f'''
#     Given a service task: {task}

#     the initial action list of solving tasks is as follows: {llm_output}

#     In a real home environment, the following environment-specific information is provided:
#     - Task-relevant objects: {all_related_objects}
#     - Their corresponding location information: {all_location_information_list}
#     - Their state information (e.g., clean, broken, opened): {state_info}

#     Please refine the initial action list based on the three-phase grounding framework:

#     **Phase 1 – Object Alignment**:  
#     Ensure that all referenced objects in the action list exist in the environment.  
#     - If an object is missing or unusable, substitute it with a suitable alternative from the available objects in {all_related_objects}.  
#     - If no appropriate replacement exists, retain the original object only if it is sufficiently functional for task execution.

#     **Phase 2 – State Verification**:  
#     Adjust the action steps to reflect the actual states of the objects.  
#     - Remove redundant actions (e.g., avoid turning on a device that is already on).  
#     - Add necessary actions if an object's current state does not satisfy preconditions for the next operation.  
#     - If an object is broken or in an unusable state, replace it or skip dependent actions accordingly.

#     **Phase 3 – Spatial Consistency**:  
#     Refine the action sequence to ensure spatial feasibility based on the locations of involved objects.  
#     - Insert necessary navigation actions (e.g., Walk) to traverse between objects located in different rooms or distant areas.  
#     - Ensure the spatial flow of actions reflects the actual layout of the environment.  
#     - Avoid spatially redundant or logically inconsistent transitions (e.g., accessing two distant objects without movement in between).

#     Strictly and clearly, output the modified action list in the format: "Step n: action", making sure all the changes reflect the correct handling of objects.
#     ----------------------------------------
#     Example of the desired output format:
#     Step 1: action 1
#     Step 2: action 2
#     Step 3: action 3
#     Step 4: action 4
#     Step 5: action 5
#     Step 6: ...
# '''

plan_grounded = get_response_strict(prompt_grounded)
print('\n' + '-'*10 + ' Grounded Actions ' + '-'*10)
print(plan_grounded)
print('-'*38)
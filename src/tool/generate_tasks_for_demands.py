import openai
import json
import os

openai.api_base = ""
openai.api_key = "" 


with open('/D2GP/json/demands.json', 'r') as f:
    demands = json.load(f)

output_file = '/D2GP/json/demand_tasks_plus.json'

def generate_task_prompt(demand):
    return (f"Demand: {demand}\n"
            "Please generate as many different tasks as possible to fulfill this demand. "
            "Each task should be a short phrase without any symbols or bullet points. "
            "For example: 'provide a chair', 'offer a seat', 'bring a cushion'. "
            "Only use plain text for each task and separate each task with a new line.")

if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    with open(output_file, 'r') as f:
        demand_tasks = json.load(f)
else:
    demand_tasks = {}

total_demands = len(demands)
completed_demands = 0

for i, demand in enumerate(demands, start=1):
    if demand in demand_tasks:
        print(f"({i}/{total_demands}) Tasks for demand '{demand}' already exist in the file.")
        completed_demands += 1
        continue
    
    prompt = generate_task_prompt(demand)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )

        if 'choices' in response and len(response['choices']) > 0:
            text_content = response['choices'][0]['message']['content']
            tasks = set(task.strip("-•* ").strip() for task in text_content.split("\n") if task.strip())
            demand_tasks[demand] = list(tasks)
            
            with open(output_file, 'w') as f:
                json.dump(demand_tasks, f, indent=4, ensure_ascii=False)
            
            completed_demands += 1
            print(f"({i}/{total_demands}) Tasks for demand '{demand}' have been saved to {output_file} - Progress: {completed_demands/total_demands:.2%}")

        else:
            print(f"No valid response for demand: {demand}")

    except openai.error.OpenAIError as e:
        print(f"OpenAI API error for demand '{demand}':", e)
    except Exception as e:
        print("Unexpected error:", e)

print(f"\nAll tasks have been processed. {completed_demands}/{total_demands} demands completed.")
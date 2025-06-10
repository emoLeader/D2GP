import openai
import json

openai.api_base = ""
openai.api_key = ""  # Your API key here

def generate_tasks_for_object(object_name):
    # Create the prompt for a single object with instruction to keep task names short
    prompt = f"""
                You are an AI assistant tasked with generating actionable tasks for a household object.

                ### Instructions:
                1. For the provided object, generate a list of all possible tasks related to that object.
                2. Tasks should be action-oriented, concise, and no longer than 5 words.
                3. Keep the task names short and to the point, like "Clean window" or "Replace filter".
                4. Consider typical household activities such as cooking, cleaning, organizing, maintenance, and recreation.
                5. Output the tasks in a simple bullet-point format without any leading dashes (-).

                ### Object Name: {object_name}

                ### Output Format:
                Task 1
                Task 2
                Task 3
            """

    # Call OpenAI API to generate the tasks for the current object
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Replace with your preferred model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        # Return the generated text (tasks)
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"


def save_tasks_to_json(object_name, tasks, output_file):
    # Load existing tasks from the JSON file
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)
    except FileNotFoundError:
        all_tasks = {}

    # Add the new tasks for the object
    all_tasks[object_name] = tasks

    # Save the updated tasks back to the file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=4, ensure_ascii=False)

    print(f"Tasks for {object_name} have been saved to {output_file}")


def main():
    # Load the list of objects
    with open('/D2GP/json/all_objects.json', 'r') as f:
        household_objects = json.load(f)
    # household_objects = ["table", "sofa[1]", "coke"]
    
    # Output file where tasks will be saved
    output_file = "household_tasks.json"

    # Generate tasks for each object one by one and save them
    for object_name in household_objects:
        print(f"Generating tasks for: {object_name}")
        
        # Generate tasks for the current object
        raw_tasks = generate_tasks_for_object(object_name)

        # Format the raw tasks into a list, remove dashes, and ensure tasks are concise
        task_list = [task.strip().lstrip("-").strip().rstrip(".") for task in raw_tasks.splitlines() if task.strip()]

        # # Further shorten task names if needed
        # task_list = [task[:20] for task in task_list]  # Limit task names to 20 characters (optional)

        # Save the tasks for this object to the JSON file
        save_tasks_to_json(object_name, task_list, output_file)


if __name__ == "__main__":
    main()
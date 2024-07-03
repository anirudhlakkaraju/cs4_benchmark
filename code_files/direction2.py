from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import openpyxl
import pandas as pd


system_prompt = """You are an English writing expert and you need to set hard essay prompts for the genre of Relistic Fiction. Your aim is to generate extremely hard constraints that make the essay writing very challenging. 

An essay prompt is defined as a main Instruction + Constraints. You will be given an Instruction as an input and you should generate more complex constraints that can be added to the Instruction which together make up the essay prompt. 

Give me a numbered list of EXACTLY 40 CONSTRAINTS.

The constraints you generate can be plot, style or format related. The constraints should be complex and creative so you may come up with constraints that can differ from the main topic of the Instruction. But keep them realistic enough and not too contradictory ensure that it's possible to write a high quality story. 

It should NOT require any specific domain knowledge to understand the constraints. DO NOT INCLUDE COMPLICATED VOCABULARY OR ASK FOR ANY POETRY RELATED FORMAT in your constriants. Make your constraints such that a flow (if any) in a story can be maintained.

The constraints should be clear and atomic, that is, if a constraint can be decomposed into multiple sub constraints, list all of them separately. DO NOT REPEAT CONSTRAINTS. JUST GIVE ONLY THE CONSTRAINT IN EACH LiST ITEM.
"""

prompt_examples = """Here are some examples with 10 constraint outputs, for your reference -

Input - Write a story that follows the journey of a professional working woman named Rachel Michelle.
Output - 
1. Rachel is single, has two kids she is supporting through high school.
2. Rachel is considering leaving her corporate job to become an entrepreneur. 
3. The story should detail the steps she takes to create a life plan and make the leap.
4. Highlight her success as an entrepreneur with five employees.
5. The book should be written in a motivational and engaging style.
6. Incorporate practical and mental wellness strategies that helped her achieve her goals.
7. Explore the complex relationship between her and her growing kids.
8. The story should be a lesson in holistic wellness and life coaching.
9. The story should cater to a young audience.
10. Write the story in 8 paragraphs or less.

Input - Write a story about the incident of two friend's day out gone wrong.
Output - 
1. One of the friends must have a secret that is revealed during the day out.
2. One of the friends must have a phobia that becomes a central theme in the story.
3. Maintain a casual tone overall but create tension when required to keep readers engaged. 
4. Involve their pet parrot in the plot. 
5. The story must contain an unexpected plot twist.
6. The friends must communicate through handwritten notes for a part of the story.
7. The story must include a mysterious stranger who impacts their day.
8. The setting of the story must change at least two times.
9. The climax of the story must involve a natural disaster.
10. The story must end with a cliffhanger that leaves the readers guessing about the fate of the friends.

"""
user_input = """
Input - {}
Output -
"""

system_prompt = system_prompt + prompt_examples

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

def chat(instruction, model="gpt-3.5-turbo", system_prompt=system_prompt, log=False):
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":system_prompt},
            {"role": "user", "content":user_input.format(instruction)},
        ]
    )

    log_usage(tokens=response.usage.total_tokens, model=model)

    if log:
        print("Total tokens used: ", response.usage.total_tokens)

    return response


def log_usage(tokens, model):
    # Get the current date and time
    current_time = datetime.now().strftime("%m-%d-%Y %H:%M:%S")

    # Write the date-time and tokens used to the file
    with open("api_usage.txt", "a") as file:
        file.write(f"{model} {current_time} : {tokens}\n")

def total_usage():
    model_tokens = {}
    model_prices = {"gpt-3.5-turbo": 0.0015, "gpt-4-turbo": 0.03}
    
    with open("api_usage.txt", "r") as file:
        for line in file:
            parts = line.split(" ")
            if len(parts) >= 4:
                model_name = parts[0]
                tokens_str = parts[-1].strip()
                tokens = int(tokens_str)
                if model_name in model_tokens:
                    model_tokens[model_name] += tokens
                else:
                    model_tokens[model_name] = tokens
    
    total_cost = 0
    for model, tokens in model_tokens.items():
        cost = (tokens * model_prices[model]) / 1000
        total_cost += cost
        print(f"Total tokens used for {model}: {tokens}")
        print(f"\nTotal cost for {model}: {cost}$")
    
    print(f"\nTotal cost so far: {total_cost}$")
    return

def get_instructions():

    # Read the Excel file into a DataFrame
    df = pd.read_excel("../Data/input_instructions.xlsx")

    instructions = [list(row.values()) for row in df.to_dict(orient='records')]
    
    return instructions

def generate_constraints(model):
    instructions = get_instructions()

    df = pd.read_excel('../Data/input_instructions.xlsx')

    dfs_to_concat = []

    i = 0
    for instruction in instructions:
        i+=1
        if i <30:
            continue
        response = chat(instruction=instruction[0], model=model, log=True)
        # print(response.choices[0].message.content)
        new_row = {
            'Instruction': instruction[0],
            'Category': instruction[1],
            'Constraints': response.choices[0].message.content
        }
        new_df = pd.DataFrame([new_row])

        dfs_to_concat.append(new_df)
        i += 1
    # Concatenate all DataFrames in the list
    df = pd.concat([df] + dfs_to_concat, ignore_index=True)

    df.to_excel('../Data/constraints_dir2_gpt4_40_final.xlsx', index=False)

generate_constraints("gpt-4-turbo")
total_usage()

story_system_prompt = """You are an expert story writer and are required to write a story based on the user provided instruction and constraints.
IMPORTANT - Satisfy all the constraints while sticking to the given word limit. Do not go beyond the word limit.
"""

def generate_stories(model="chatgpt-3.5-turbo"):
    df = pd.read_excel('../Data/constraints_dir2_gpt4_40_final2.xlsx')

    i = 0
    for index, row in df.iterrows():
        user_prompt = f"Instruction - {row['instruction']}\nConstraints - {row['constraints']}"
        
        # Feed user prompt and system prompt into chat method
        response = client.chat.completions.create(
            model= model,
            messages=[
                {"role": "system", "content":story_system_prompt},
                {"role": "user", "content":user_prompt},
            ]
        )
        
        # Extract generated story from response
        generated_story = response.choices[0].message.content
        
        # Append generated story to the DataFrame
        df.at[index, 'final_prompt'] = user_prompt
        df.at[index, 'generated_story'] = generated_story

    df.to_excel('../Data/constraints_dir2_gpt4_40_final2.xlsx', index=False)


generate_stories("gpt-4-turbo")



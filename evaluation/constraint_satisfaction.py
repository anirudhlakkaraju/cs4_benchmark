import os
import pandas as pd
from openai import OpenAI
# from dotenv import load_dotenv
import argparse

def read_api_key(api_key_file):
    with open(api_key_file, "r") as file:
        api_key = file.read().strip()
    return api_key

# def generate_final_prompt(row):
#     # Extracting constraints
#     story = row["FinalGeneratedStory"]
#     constraints = row["SelectedConstraints"]
#     no_of_constraints = row['Number_of_Constraints']
#     # Combining constraints with story
#     final_prompt = f"""Input - \nStory: - {story}\n \nNumber of Constraints in the story: - {no_of_constraints}\nConstraints: - \n{constraints} \n\n Output - Give me Number of Constraints Satisfied"""
#     return final_prompt

def generate_final_prompt(row):
    # Extracting constraints
    story = row["FinalGeneratedStory"]
    constraints = row["SelectedConstraints"]
    # Combining constraints with story
    final_prompt = f"""Story: {story}\nConstraints: \n{constraints}\n"""
    return final_prompt

def add_final_prompt_column(df):
    df["CS_FinalPrompt"] = df.apply(generate_final_prompt, axis=1)
    return df

def read_system_prompt(system_prompt_file):
    with open(system_prompt_file, "r") as file:
        system_prompt = file.read()
    return system_prompt

def chat_with_openai(user_prompt, system_prompt, model="gpt-3.5-turbo-0125", api_key=None):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content

import os

def save_dataframe(df, save_path, index=False):
    # Extract the directory and filename from the save_path
    directory, filename = os.path.split(save_path)
    
    # Create the directory if it doesn't exist
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(save_path, index=index)

# Define the number of rows after which to save the DataFrame

def process_responses(file_path, save_path, system_prompt_file, model, api_key_file):
    df = pd.read_csv(file_path)
    # print(df.info())
    df = add_final_prompt_column(df)
    # print(df.info())
    # print(system_prompt)
    system_prompt = read_system_prompt(system_prompt_file)
    # print(system_prompt)
    print("**"*10)
    # print(api_key_file)
    api_key = read_api_key(api_key_file)
    # print(api_key)

    save_interval = 20

    # Define the folder where the partial save files will be stored
    partial_save_folder = "edited_prompt_run_partial_saves"

    # Extract the base path from final_save_path
    base_path = os.path.dirname(save_path)

    # Create the partial saves folder if it doesn't exist
    partial_save_folder_path = os.path.join(base_path, partial_save_folder)
    os.makedirs(partial_save_folder_path, exist_ok=True)

    for index, row in df.iterrows():
        # print(row)
        try:
            response_content = chat_with_openai(row["CS_FinalPrompt"], system_prompt, model, api_key=api_key,)
            df.at[index, f'CS_{model}_Response'] = response_content
            if (index + 1) % save_interval == 0:
                partial_save_path = os.path.basename(file_path).replace(".csv", f"_partial_{index + 1}.csv")
                print("Partial Saving the file till:", index, partial_save_path)
                save_dataframe(df.iloc[:index + 1], os.path.join(partial_save_folder_path, partial_save_path))
        except Exception as e:
            print(f"Error occurred at row {index}: {e}, Prompt: {row['CS_FinalPrompt']}")
            df.at[index, "CS_FinalPrompt"] = "ERROR"
    
    print("File are partially saved at path:", partial_save_folder_path)
    print("Final file path:", save_path)
# Save the final DataFrame
    save_dataframe(df, save_path)
        
    # df.to_csv(save_path, index=False)

import os

def process_folder(folder_path, save_folder, system_prompt_file, direction, model, api_key_file):
    try:
        files = os.listdir(folder_path)
        print(files)
        startswith = ""
        endswith = ""
        if direction == "d2":
            startswith = "d2_olmo_basehf"
            endswith = "_d2.csv"
        elif direction == "d3":
            # startswith = "d3_"
            startswith = "d3_"
            endswith = "_d3.csv"
        else:
            raise ValueError("Invalid direction. Direction must be 'd2' or 'd3'.")
        # print(startswith, endswith)
        # vllm_trials/Expansion/direction3/olmo_storygen/parsed/d3_olmo_basehf_d3.csv
        target_files = [file for file in files if file.startswith(startswith) and file.endswith(endswith)]
        print("Target Files", target_files)
        for file in target_files:
            print("*"*50)
            file_path = os.path.join(folder_path, file)

            # save_folder = os.path.join(folder_path, f'responses_{model}')
            os.makedirs(save_folder, exist_ok=True)
            final_save_path = os.path.join(save_folder, file.replace("batch_storygens", model).replace('.csv', '_cs_output.csv'))
            # print(file_path)
            # print(api_key_file)
            process_responses(file_path, final_save_path, system_prompt_file, model, api_key_file)
            print("File Processed:", file)
            print("File Saved at:", final_save_path)
            print("*"*50)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files with OpenAI chat.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing input CSV files")
    parser.add_argument("--save_folder", type=str, help="Path to the save output of CSV files")
    parser.add_argument("--system_prompt_file", type=str, help="Path to the text file containing the system prompt")
    parser.add_argument("--direction", type=str, help="Direction 2 or 3")
    parser.add_argument("--api_key_file", type=str, help="Path to the text file containing the OpenAI API key")
    parser.add_argument("--model", type=str, help="Model to evaluate gpt-4-turbo, gpt-4, and gpt-3.5-turbo. Default gpt-4")
    args = parser.parse_args()
    print("Arguments Passed", args)
    process_folder(args.folder_path, args.save_folder, args.system_prompt_file, args.direction, args.model, args.api_key_file)
    # python script.py --folder_path "vllm_trials/Expansion/direction2" --system_prompt_file "system_prompt.txt"  --direction "d2" --api_key_file "/Users/rohithsiddharthareddy/Library/CloudStorage/OneDrive-UniversityofMassachusetts/Acads/Spring 24/696DS/vllm_trials/code_trials/api_key.txt" --model "gpt-4"

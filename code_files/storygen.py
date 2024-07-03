import argparse
from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch
import re
import os
from tqdm import tqdm 
import subprocess
import gc


max_tokens = 4096
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)


def generate_response(tokenizer, olmo, prompt_text, max_tokens=4096):
    # Load the model and tokenizer
    
    # Create chat history
    chat = [
        {"role": "user", "content": prompt_text},
    ]

    # Apply chat template and tokenize
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to('cuda')

    # Generate response
    response = olmo.generate(input_ids=inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
    inputs = inputs.cpu()

    # Decode and return response
    return tokenizer.batch_decode(response, skip_special_tokens=True)[0]

    
def clear_cache_if_needed(directory, directo):
    # Execute the df command to check disk usage for the specified directory
    abs_directory = os.path.expanduser(directory)

    df_output = subprocess.check_output(['df', '-h', abs_directory]).decode('utf-8')
    # Split the output lines
    df_lines = df_output.split('\n')
    # Get the line containing usage information
    usage_line = df_lines[1]
    # Split the line into fields
    fields = usage_line.split()
    # Extract the usage percentage
    usage_percentage = int(fields[4].rstrip('%'))
    
    # Check if usage exceeds 85%
    files_in_directory = os.listdir('/home/rbheemreddy_umass_edu/.cache/huggingface/hub')
    print(files_in_directory)

    if usage_percentage>60:
        command = f"rm -rfv ~/.cache/huggingface/hub/*"
        subprocess.call(command, shell=True)

        print(f"Cache cleared successfully for models as usage percentage is {usage_percentage}")
    else:
        print("Disk usage is below 60%. No need to clear cache. Usage Percentage:", usage_percentage)


def clear_cuda_memory():
    torch.cuda.empty_cache()


"""Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
def addNewStory(df, list_num_constraints, llm=None):
    # Initialize an empty DataFrame to store the results
    single_instruction_df = pd.DataFrame(columns=['Instruction', 'Constraints', 'BaseStory', 'Direction', 'Model', 'SelectedConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])


    if llm==None:
        olmo = OLMoForCausalLM.from_pretrained(model).to('cuda')
        tokenizer = OLMoTokenizerFast.from_pretrained(model)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):

        revision_prompt = f"Now revise the given BaseStory to satisfy the following constraints within 500 words: \n{row['SelectedConstraints']}"
        storygen_prompt = f"""Story Instruction: {row['Instruction']}\nBaseStory: {row["BaseStory"]}\nTask: {revision_prompt}"""
        
        if llm==None:
            final_generated_story = generate_response(tokenizer, olmo, storygen_prompt, max_tokens)
        else:
            # sampling_params = 
            output2 = llm.generate([storygen_prompt], sampling_params)
            for output in output2:
                final_generated_story = output.outputs[0].text

        # Add the data to the result DataFrame

        single_instruction_df.loc[len(single_instruction_df)] = {
            'Instruction': row['Instruction'],
            # 'Category': row['Category'],
            'Constraints': row['Constraints'],
            'BaseStory': row["BaseStory"],
            'Direction': row['Direction'],
            'Model': row["Model"],
            'SelectedConstraints': row['SelectedConstraints'],
            'Number_of_Constraints': row['Number_of_Constraints'],
            'Final_Prompt': storygen_prompt,
            'FinalGeneratedStory': final_generated_story
        }

    return single_instruction_df

def generalcall(llm, name_model, filename):

    if "gemma" in name_model:
        base_path = 'gemma'
    if "Llama" in name_model:
        base_path = 'llama'
    if "Mistral" in name_model:
        base_path = 'mistral'
    if "OLMo-7B-hf" in name_model:
        base_path = 'olmo_basehf'
    if "OLMo-7B-SFT" in name_model:
        base_path = 'olmo_sft'
    if "OLMo-7B-Instruct" in name_model:
        base_path = 'olmo_instruct'

    auto_gen = pd.read_csv(filename)
    unique_instructions = auto_gen['Instruction'].unique()

    auto_gen_eval = auto_gen[auto_gen['Instruction'].isin(unique_instructions)].copy()

    # Add new columns to store outputs
    auto_gen_eval['Final_Prompt'] = ''
    auto_gen_eval['FinalGeneratedStory'] = ''
    auto_gen_eval['Model'] = base_path

    # List of constraints to try
    list_num_constraints = [7, 15, 23, 31, 39]

    # Initialize an empty list to store all generated DataFrames
    all_dfs = []
    count=0

    combined_df = addNewStory(auto_gen_eval, list_num_constraints, llm)

    # Append the generated DataFrame to the list
    all_dfs.append(combined_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    total_stories_df = pd.concat(all_dfs, ignore_index=True)

    # Save the combined DataFrame to a single CSV file
    if "direction3" in filename:
        d = "d3"
    elif "direction2" in filename:
        d = 'd2'

    save_path = f"{base_path}"
    os.makedirs(base_path, exist_ok=True)
    print("Path saving file:", save_path)
    total_stories_df.to_csv(os.path.join(save_path, f'{d}_{base_path}_{d}.csv'), index=False)


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Generate Stories")

    # Add arguments to the parser
    parser.add_argument('file_path', type=str, help='The path to the constraints')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    file_path = args.file_path

    for model  in ['allenai/OLMo-7B-SFT', 'allenai/OLMo-7B-Instruct', 'allenai/OLMo-7B-hf']:
        
        clear_cache_if_needed(directory)
        
        if model in ['allenai/OLMo-7B-hf']:
            print("name of model", model)
            llm = LLM(model=model, dtype=torch.float16)
            generalcall(llm=llm, name_model=model, filename=file_path)
            del llm
        else:
            generalcall(llm=None, name_model=model, filename=file_path)

        print(f"Model {model} DONE")
        
        # Clear the model object from memory
        del model
        
        gc.collect()
        clear_cuda_memory()
        
        print("CUDA memory cleared.")

        clear_cache_if_needed(directory)


if __name__=="__main__":
    main()



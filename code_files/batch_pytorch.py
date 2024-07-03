from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch
import re
import os
from tqdm import tqdm  # Import tqdm for progress tracking
import subprocess

max_tokens = 4096
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)

    
def clear_cache_if_needed(directory):
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

    
# Example usage:
directory = "~"  # Specify the directory you want to check (e.g., "~" for the home directory)


def clear_cuda_memory():
    torch.cuda.empty_cache()
# from transformers import OLMoForCausalLM, OLMoTokenizerFast

# def generate_response(tokenizer, olmo, prompt_text, max_tokens=4096):
#     # Load the model and tokenizer
    

#     # Create chat history
#     chat = [
#         {"role": "user", "content": prompt_text},
#     ]

#     # Apply chat template and tokenize
#     prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to('cuda')

#     # Generate response
#     # response = olmo.generate(input_ids=inputs.to(olmo.device), max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
#     response = olmo.generate(input_ids=inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
#     inputs = inputs.cpu()
#     # Decode and return response
#     return tokenizer.batch_decode(response, skip_special_tokens=True)[0]




# Create an LLM.

# def addNewStory(df, list_num_constraints, llm=None):
#     """Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
#     # Initialize an empty DataFrame to store the results
#     single_instruction_df = pd.DataFrame(columns=['Instruction', 'Constraints', 'BaseStory', 'Direction', 'Model', 'SelectedConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])


#     if llm==None:
#         olmo = OLMoForCausalLM.from_pretrained(model).to('cuda')
#         tokenizer = OLMoTokenizerFast.from_pretrained(model)
#     for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
#         # print("Processing row number:", index)

#         prompt2_start = f"Now modify the existing story to accommodate the following constraints: {row['SelectedConstraints']} into the LLM generated story and come up with a new story in 500 words: "
#         final_prompt = f"""User: "  {row['Instruction']}" \n BaseStory: " {row["BaseStory"]} " \n User Instruction: " {prompt2_start} """
        
#         if llm==None:
#             final_generated_story = generate_response(tokenizer, olmo, final_prompt, max_tokens)
#         else:
#             # sampling_params = 
#             output2 = llm.generate([final_prompt], sampling_params)
#             for output in output2:
#                 final_generated_story = output.outputs[0].text

#         # Add the data to the result DataFrame

#         single_instruction_df.loc[len(single_instruction_df)] = {
#             'Instruction': row['Instruction'],
#             # 'Category': row['Category'],
#             'Constraints': row['Constraints'],
#             'BaseStory': row["BaseStory"],
#             'Direction': row['Direction'],
#             'Model': row["Model"],
#             'SelectedConstraints': row['SelectedConstraints'],
#             'Number_of_Constraints': row['Number_of_Constraints'],
#             'Final_Prompt': final_prompt,
#             'FinalGeneratedStory': final_generated_story
#         }

#     return single_instruction_df


"""""








"""

def generate_responses(tokenizer, olmo, prompt_texts, max_tokens=4096):
    # Create chat history for each prompt text
    print("Calling generate_responses")
    print("leng of prompt_texts", len(prompt_texts))
    chats = [{"role": "user", "content": text} for text in prompt_texts]
    print("*****"*10)
    print("Length of chats in generate_response", len(chats))
    # Apply chat template and tokenize for each prompt
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
    inputs = tokenizer.batch_encode_plus(prompts, add_special_tokens=False, return_tensors="pt").to('cuda')

    # Generate responses
    responses = olmo.generate(input_ids=inputs.input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
    inputs = inputs.input_ids.cpu()
    print(responses)
    print("*****"*10)

    output = [tokenizer.batch_decode(response, skip_special_tokens=True) for response in responses]
    # Decode and return responses
    return output

def generate_response(tokenizer, olmo, prompt_text, max_tokens=4096):
    # Call the generate_responses function for batch inference
    responses = generate_responses(tokenizer, olmo, [prompt_text], max_tokens)
    print("Length of responses in generate_response", len(responses))
    return responses[0]


def addNewStory(df, list_num_constraints, llm=None, batch_size=8):
    """Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
    # Initialize an empty DataFrame to store the results
    single_instruction_df = pd.DataFrame(columns=['Instruction', 'Constraints', 'BaseStory', 'Direction', 'Model', 'SelectedConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])

    if llm is None:
        olmo = OLMoForCausalLM.from_pretrained(model).to('cuda')
        tokenizer = OLMoTokenizerFast.from_pretrained(model)
    else:
        olmo = llm

    batch_instructions = []
    generated_stories = []
    instructions = []
    print("Dataset info", df.info())
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        prompt2_start = f"Now modify the existing story to accommodate the following constraints: {row['SelectedConstraints']} into the LLM generated story and come up with a new story in 500 words: "
        final_prompt = f"""User: "  {row['Instruction']}" \n BaseStory: " {row['BaseStory']} " \n User Instruction: " {prompt2_start} """

        batch_instructions.append(final_prompt)
        instructions.append(final_prompt)
        print("Length of batch", len(batch_instructions))
        print("Length of instructions", len(instructions))

        if len(batch_instructions) == batch_size:
            print("Value of llm", llm)
            if llm is None:
                output2 = generate_response(tokenizer, olmo, batch_instructions)
                print(output2, '\n', type(output2))
                generated_stories.extend(generate_response(tokenizer, olmo, batch_instructions))
            else:
                output2 = llm.generate(batch_instructions)
                generated_stories.append(output.outputs[0].text for output in output2)
            batch_instructions = []

    # Process the remaining prompts
    if batch_instructions:
        if llm is None:
            generated_stories += generate_response(tokenizer, olmo, batch_instructions)
        else:
            output2 = llm.generate(batch_instructions)
            generated_stories += [output.outputs[0].text for output in output2]
    print(f"Length of generated stores {len(generated_stories)}")
    print(f"Length of instruction {len(instructions)}")
    # Add the data to the result DataFrame
    for i, row in enumerate(df.itertuples()):
        single_instruction_df.loc[i] = {
            'Instruction': row.Instruction,
            'Constraints': row.Constraints,
            'BaseStory': row.BaseStory,
            'Direction': row.Direction,
            'Model': row.Model,
            'SelectedConstraints': row.SelectedConstraints,
            'Number_of_Constraints': row.Number_of_Constraints,
            'Final_Prompt': instructions[i],
            'FinalGeneratedStory': generated_stories[i]
        }

    return single_instruction_df



#RUNNING FOR DIRECTION3 TABLE

# Model Definition
# model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
# model_name = 'google/gemma-7b-it'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'

# -hf models only work as they have config.json
# model_name = 'meta-llama/Llama-2-7b'
# model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = 'meta-llama/Llama-2-7b-chat'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'



def generalcall(llm, name_model):
    # filename = "/home/rbheemreddy_umass_edu/vllm_trials/direction3/df3_gpt_12_50_selected_constraints.csv"

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

    filename = "/home/rbheemreddy_umass_edu/vllm_trials/Expansion/direction2/d2_with_constraints.csv"
    auto_gen = pd.read_csv(filename)[:16]
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
    print(total_stories_df.info())
    # Save the combined DataFrame to a single CSV file

    if "direction3" in filename:
        d = "d3"
    elif "direction2" in filename:
        d = 'd2'
    # save_path = f"{base_path}/{d}_overnight_storygens_{base_path}_{d}.csv"
    save_path = f"{base_path}"
    os.makedirs(base_path, exist_ok=True)
    print("Path saving file:", save_path)
    total_stories_df.to_csv(os.path.join(save_path, f'{d}_{base_path}_{d}.csv'), index=False)

# for model  in ['mistralai/Mistral-7B-Instruct-v0.1', 'google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf']:
for model  in ['allenai/OLMo-7B-SFT',  'allenai/OLMo-7B-Instruct', 'allenai/OLMo-7B-hf']:
# for model  in [ 'allenai/OLMo-7B-SFT',  'allenai/OLMo-7B-Instruct']:
# for m  in []:
    clear_cache_if_needed(directory)
    
    if model in ['allenai/OLMo-7B-hf']:
        print("name of model", model)
        llm = LLM(model=model, dtype=torch.float16)
        generalcall(llm=llm, name_model=model)
        del llm
    else:
        generalcall(llm=None, name_model=model)

    print(f"Model {model} DONE")
    # Clear the model object from memory
    del model
    import gc
    gc.collect()
    clear_cuda_memory()
    print("CUDA memory cleared.")

    clear_cache_if_needed(directory)



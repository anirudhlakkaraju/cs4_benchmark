import pandas as pd

# Read the CSV file
df = pd.read_csv("/home/rbheemreddy_umass_edu/vllm_trials/Expansion/direction3/olmo_storygen/original/d3_olmo_sft_d3.csv")
print(df.info())
# Define your splitting words
spl_word_assistant = "<|assistant|>"
spl_word_user = "<|user|>"

# Function to parse the story
def parse_story(story_string):
    res_instruction = story_string[story_string.find(spl_word_user) + len(spl_word_user): story_string.find(spl_word_assistant)]
    res_output = story_string[story_string.find(spl_word_assistant) + len(spl_word_assistant): ]
    return res_output

# Apply the function to the FinalGeneratedStory column
df['FinalGeneratedStory'] = df['Model_Response'].apply(parse_story)
print(df.info())
# Save the updated DataFrame to a new CSV file
df.to_csv("/home/rbheemreddy_umass_edu/vllm_trials/Expansion/direction3/olmo_storygen/parsed/d3_olmo_sft_d3.csv", index=False)

import argparse
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime

# Initialize OpenAI client
def initialize_openai(api_key):
    return OpenAI(api_key=api_key)

# Chat function to send prompt to OpenAI API
def chat(client, instruction, model="gpt-3.5-turbo", system_prompt=""):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
    )

    return response

# Parse the evaluation results
def parse_evaluation(evaluation):
    if pd.isna(evaluation):
        return None
    try:
        lines = [line.strip() for line in evaluation.split('\n') if line.strip()]
        parsed = {
            'grammar_score_A': float(lines[1].split('-')[1].split('/')[0].strip()),
            'grammar_score_B': float(lines[2].split('-')[1].split('/')[0].strip()),
            'coherence_score_A': float(lines[4].split('-')[1].split('/')[0].strip()),
            'coherence_score_B': float(lines[5].split('-')[1].split('/')[0].strip()),
            'likability_score_A': float(lines[7].split('-')[1].split('/')[0].strip()),
            'likability_score_B': float(lines[8].split('-')[1].split('/')[0].strip()),
            'grammar_pref': lines[0].split(': ')[1],
            'coherence_pref': lines[3].split(': ')[1],
            'likability_pref': lines[6].split(': ')[1],
            'overall_pref': lines[9].split(': ')[1]
        }
        a = 0
        b = 0
        if parsed['grammar_score_A'] >= parsed['grammar_score_B']:
            parsed['grammar_pref'] = "A"
            a += 1
        else:
            parsed['grammar_pref'] = "B"
            b += 1

        if parsed['coherence_score_A'] >= parsed['coherence_score_B']:
            parsed['coherence_pref'] = "A"
            a += 1
        else:
            parsed['coherence_pref'] = "B"
            b += 1

        if parsed['likability_score_A'] >= parsed['likability_score_B']:
            parsed['likability_pref'] = "A"
            a += 1
        else:
            parsed['likability_pref'] = "B"
            b += 1

        if a >= b:
            parsed['overall_pref'] = "A"
        else:
            parsed['overall_pref'] = "B"

        return parsed
    except Exception as e:
        print(e)
        return None

def pairwise_eval(client, story1, story2, model="gpt-3.5-turbo"):
    # Prompts
    system_prompt1 = """
    You are an English writing expert and you can compare and evaluate story essays on these metrics with the following definitions -
        1. Grammar: Which story has better writing and grammar comparitively?
        2. Coherence: Which story has a better logical flow and the writing fits together with respect to the plot?
        3. Likability: Which story do you find more enjoyable to read?
    You will be given two Stories - Story A and Story B.
    Add a rating out of 5 for each category, specify which story you prefer for each metric by responding with just the letter "A" or "B" followed by a hyphen and one line reasoning for your preference.
    For each category provide a category winner story as the letter "A" or "B", based on the category ratings.
    Finally, assign an overall winner story as the letter "A" or "B" based on the ratings and category wins.

    IMPORTANT - DO NOT GIVE ANY OTHER TEXT APART FROM THE SCORE, METRICS AND PREFERENCE. FOLLOW THE EXACT FORMAT AS GIVEN IN THE FOLLOWING EXAMPLES.

    EXAMPLE OUTPUT 1:
    Grammar  Preference: A
    A - 5/5: Story A has a few minor grammatical issues, but overall, it demonstrates strong control of language.
    B - 4/5: Story B is well-written but has slightly more noticeable issues in grammar and sentence structure.
    Coherence  Preference: A
    A - 4.5/5: Story B has a strong coherence, effectively conveying the emotional journey and the progression of events.
    B - 4/5: Story A maintains a consistent and engaging narrative flow, though some parts are a bit abstract.
    Likability  Preference: A
    A - 4/5: Story B's realistic and emotional narrative is likely to resonate more with a wide range of readers.
    B - 3.5/5: Story A is imaginative and intriguing, but its abstract nature might not appeal to all readers.
    Overall Winner: A

    EXAMPLE OUTPUT 2:
    Grammar Preference: B
    A - 3/5: Story A has some complex sentences that are difficult to follow, with occasional grammatical errors.
    B - 4/5: Story B is well-written with minor grammatical mistakes and clear sentence structures.
    Coherence Preference: B
    A - 2/5: The plot of Story A is somewhat confusing and disjointed, especially with the sudden introduction of an old sage.
    B - 5/5: Story B maintains a coherent narrative, with each event logically building on the previous one, enhancing the story’s flow.
    Likability Preference: B
    A - 3/5: Story A is heartfelt but its erratic narrative structure detracts from its overall appeal.
    B - 5/5: Story B is compelling and maintains consistent character development, making it more enjoyable and engaging.
    Overall Winner: B

    """

    prompt0 = f"""
    Story A:
    {story1}

    Story B:
    {story2}
    """
    response1 = chat(client, prompt0, model=model, system_prompt=system_prompt1)
    return response1.choices[0].message.content

# Evaluate stories and save results
def evaluate_stories(grouped_dfs, client, output_dir, max_trials=35, max_redo=3):
    count = 0
    for instruction, df in grouped_dfs.items():
        count += 1
        if count > max_trials:
            continue
        
        row_with_11 = df[df['Number_of_Constraints'] == 23].iloc[1]
        for index, other_row in df.iterrows():
            if other_row['story_id'] == row_with_11['story_id']:
                continue
            
            rand_trial = np.random.randint(2)
            story_a, story_b = (other_row, row_with_11) if rand_trial else (row_with_11, other_row)
            results = None
            needs_parsing = 0
            
            for attempt in range(max_redo):
                try:
                    results = pairwise_eval(client, story_a['FinalGeneratedStory'], story_b['FinalGeneratedStory'])
                    parsed_results = parse_evaluation(results)
                    if parsed_results:
                        for key, value in parsed_results.items():
                            df.loc[other_row.name, key] = value
                        df.loc[other_row.name, 'order'] = rand_trial
                        needs_parsing = 0
                        break
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    needs_parsing = 1
                    break

            df.loc[other_row.name, 'needs_parsing'] = needs_parsing
            df.loc[other_row.name, 'evaluations'] = results
        
        print(f"Evaluations complete for instruction {instruction}")

        # Save the DataFrame to CSV
        output_file = os.path.join(output_dir, f"{instruction}_evaluations.csv")
        df.to_csv(output_file, index=False)
        grouped_dfs[instruction] = df

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description="Run story evaluation using OpenAI API")
    
    # API key and input/output paths
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--input_file", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--max_trials", type=int, default=35, help="Maximum number of trials for evaluation")
    
    args = parser.parse_args()

    # Initialize OpenAI API client
    client = initialize_openai(args.api_key)

    # Load the input file into a pandas DataFrame
    df = pd.read_csv(args.input_file)
    
    # Group the DataFrame by instruction if needed
    grouped_dfs = {"default": df}  # In case grouping is needed, adapt this based on your use case
    
    # Evaluate the stories and save results
    evaluate_stories(grouped_dfs, client, args.output_dir, max_trials=args.max_trials)

if __name__ == "__main__":
    main()


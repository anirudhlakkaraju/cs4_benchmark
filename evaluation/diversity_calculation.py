# Setup NLTK
import string
import re
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd
nltk.download('punkt')

# This code takes a CSV file as an input (path to CSV). The CSV file has three columns - Story1, Story2, Story3 representing three stories generated for each input prompt in CS4 benchmark. 
# The code then computes the unique and total number of 2, 3, and 4 grams in each of the stories, and then computes the overall diversity of the stories for each prompt. 
# The code finally stores a CSV (file path to be given) that has columns containing the unique and total number of n-grams along with the diversity scores. 

def main():
    # Function to convert text to lowercase and remove punctuation
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text

    # Function to calculate n-gram statistics
    def ngram_statistics(text):
        tokens = word_tokenize(text)
        results = []
        for n in range(2, 5):  # For 2-grams, 3-grams, and 4-grams
            generated_ngrams = list(ngrams(tokens, n))
            unique_ngrams = len(set(generated_ngrams))
            total_ngrams = len(generated_ngrams)
            results.append(unique_ngrams)
            results.append(total_ngrams)
        return results

    
    # Data manipulation
    path = "/path/to/your/dataset.csv"  # Adjust path accordingly
    df = pd.read_csv(path)

    # Compute n-gram statistics for each story and calculate diversity indices
    for story_label in ['Story1', 'Story2', 'Story3']:
        for i in range(len(df)):
            story = df.loc[i, story_label]
            text = preprocess_text(story)
            n_grams = ngram_statistics(text)
            df.loc[i, f"{story_label}_unique 2-grams"] = n_grams[0]
            df.loc[i, f"{story_label}_total 2-grams"] = n_grams[1]
            df.loc[i, f"{story_label}_unique 3-grams"] = n_grams[2]
            df.loc[i, f"{story_label}_total 3-grams"] = n_grams[3]
            df.loc[i, f"{story_label}_unique 4-grams"] = n_grams[4]
            df.loc[i, f"{story_label}_total 4-grams"] = n_grams[5]

    # Calculate aggregated diversity scores
    for n in ['2', '3', '4']:
        df[f'Sum_{n}Grams'] = df[f'Story1_unique {n}-grams'] + df[f'Story2_unique {n}-grams'] + df[f'Story3_unique {n}-grams']
        df[f'Total_{n}Grams'] = df[f'Story1_total {n}-grams'] + df[f'Story2_total {n}-grams'] + df[f'Story3_total {n}-grams']
        df[f'Diversity_{n}G'] = df[f'Sum_{n}G'] / df[f'Total_{n}G']

    df["Product_diversity"] = df['Diversity_2G'] * df['Diversity_3G'] * df['Diversity_4G']
    df.to_csv(path, index=False)

if __name__ == "__main__":
    main()

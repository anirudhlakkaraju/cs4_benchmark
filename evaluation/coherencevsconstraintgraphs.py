import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os

# Dictionary mapping model short names to full names
model_dict = {
    "gemma": "Gemma-7B Instruct",
    "mistral": "Mistral-7B Instruct",
    "llama": "Llama-2-7B Chat",
    "olmo_basehf": "OLMo Base",
    "olmo_sft": "OLMo SFT",
    "olmo_instruct": "OLMo Instruct"
}

# Function to annotate points on the plot
def annotate_points(x, y, labels, color):
    for i, label in enumerate(labels):
        annotation_text = f"Constraints: {label}"
        offset_x = random.randint(-15, 15)
        offset_y = random.randint(-15, 15)
        plt.annotate(annotation_text, (x[i], y[i]),
                     textcoords="offset points",
                     xytext=(offset_x, 5 + offset_y),
                     ha='center',
                     fontsize=12,
                     fontstyle='italic',
                     color=color)

# Function to process data and generate the plot
def process_and_plot_normalized(grouped_results, title_suffix, output_dir, save_as_pdf=False, pdf_filename="plot.pdf"):
    sns.set(style="whitegrid")

    # Set colors based on available models
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if 'gemma' in grouped_results else ['tab:red', 'tab:purple', 'tab:brown']

    plt.figure(figsize=(12, 7))
    idx = 0
    for model, grouped_model_df in grouped_results.items():
        plt.plot(grouped_model_df['average_percentage_gpt4'], grouped_model_df['normalized_coherence_score'],
                 label=model_dict.get(model, model), marker='o', color=colors[idx])
        annotate_points(grouped_model_df['average_percentage_gpt4'].values, 
                        grouped_model_df['normalized_coherence_score'].values, 
                        grouped_model_df.index, colors[idx])
        idx += 1

    plt.xlabel('Constraint Satisfaction (%)', fontsize=14)
    plt.ylabel('Normalized Coherence Score', fontsize=14)
    plt.legend(title='Model', fontsize=11, title_fontsize='14')
    plt.grid(True)

    if save_as_pdf:
        pdf_path = os.path.join(output_dir, pdf_filename)
        plt.savefig(pdf_path, format='pdf')
        print(f"Plot saved as {pdf_path}")
    else:
        plt.show()

# Main function to load data and generate the plot
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot Normalized Coherence Score by Constraint Satisfaction")
    
    # Arguments for input CSV files and output directory
    parser.add_argument("--input_csv", nargs='+', required=True, help="List of CSV files for each model (format: model_name file_path).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output plot.")
    parser.add_argument("--save_as_pdf", action="store_true", help="Flag to save the plot as PDF instead of displaying it.")
    
    args = parser.parse_args()

    # Parse input CSV files
    file_dict = {args.input_csv[i]: args.input_csv[i + 1] for i in range(0, len(args.input_csv), 2)}

    # Load data from CSV files into grouped DataFrames
    grouped_results = {}
    for model, file_path in file_dict.items():
        df = pd.read_csv(file_path)
        grouped_model_df = df.groupby('Number_of_Constraints').agg(
            total_coherence_score=('coherence_score', 'sum'),
            total_percentage_gpt4=('Percentage_GPT4', 'sum')
        )
        grouped_model_df['average_percentage_gpt4'] = grouped_model_df['total_percentage_gpt4'] / df['Number_of_Constraints'].value_counts()
        max_coherence_score = grouped_model_df['total_coherence_score'].max()
        grouped_model_df['normalized_coherence_score'] = grouped_model_df['total_coherence_score'] / max_coherence_score
        grouped_results[model] = grouped_model_df

    # Generate and save or display the plot
    process_and_plot_normalized(grouped_results, "Coherence vs Constraints", args.output_dir, args.save_as_pdf, "coherence_vs_constraints.pdf")

if __name__ == "__main__":
    main()





"""# Quality Under Constraints

QUC_n = (Normalized coherence score given n constraints) * (constraint satisfaction percentage given n constraints).
"""

def calculate_quc_and_rcs(grouped_results):
    quc_results = {}
    rcs_results = {}

    for model, df in grouped_results.items():
        quc = df['normalized_coherence_score'] * df['average_percentage_gpt4']
        quc_results[model] = quc

        rcs = {}
        constraints = df.index.tolist()
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                m = constraints[i]
                n = constraints[j]
                rcs[f"{m}-{n}"] = quc.loc[m] - quc.loc[n]

        rcs_results[model] = rcs

    return quc_results, rcs_results


def Quc_9VsRcs_7_39(quc_results, rcs_results):
    # Initialize an empty list to store the comparison data
    comparison_data = []

    for model in quc_results.keys():
        # Extract QUC_39 for the model
        quc_39 = quc_results[model].get("39", None)

        # Extract RCS_7-39 for the model
        rcs_7_39 = rcs_results[model].get('39-7', None)

        # Append the data to the comparison list
        comparison_data.append({
            'Model': model,
            'QUC_39': quc_39,
            'RCS_7-39': rcs_7_39
        })

    # Create a DataFrame to display the comparison table
    return pd.DataFrame(comparison_data)

d3_gpt4_mgl_grouped = load_grouped_dfs_from_json("d3_gpt4_mgl_grouped.json")
d3_gpt4_mgl_quc, d3_gpt4_mgl_rcs = calculate_quc_and_rcs(d3_gpt4_mgl_grouped)
comparison_df = Quc_9VsRcs_7_39(d3_gpt4_mgl_quc, d3_gpt4_mgl_rcs)
comparison_df

d3_mgl_grouped = load_grouped_dfs_from_json("d3_mgl_grouped.json")
d3_mgl_quc, d3_mgl_rcs = calculate_quc_and_rcs(d3_mgl_grouped)
comparison_df = Quc_9VsRcs_7_39(d3_mgl_quc, d3_mgl_rcs)
comparison_df

d2_mgl_grouped = load_grouped_dfs_from_json("d2_mgl_grouped.json")
d2_mgl_quc, d2_mgl_rcs = calculate_quc_and_rcs(d2_mgl_grouped)
comparison_df = Quc_9VsRcs_7_39(d2_mgl_quc, d2_mgl_rcs)
comparison_df

d3_olmo_grouped = load_grouped_dfs_from_json("d3_olmo_grouped.json")
d3_olmo_quc, d3_olmo_rcs = calculate_quc_and_rcs(d3_olmo_grouped)
comparison_df = Quc_9VsRcs_7_39(d3_olmo_quc, d3_olmo_rcs)
comparison_df

d2_olmo_grouped = load_grouped_dfs_from_json("d2_olmo_grouped.json")
d2_olmo_quc, d2_olmo_rcs = calculate_quc_and_rcs(d2_olmo_grouped)
comparison_df = Quc_9VsRcs_7_39(d2_olmo_quc, d2_olmo_rcs)
comparison_df

"""# Relative Creativity Score m n

RCS_{m-n} = QUC_m - QUC_n
"""












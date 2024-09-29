import logging
import argparse
import os
import subprocess
import sys

# Set up logging
logging.basicConfig(
    filename='eval_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_script(script_name, args):
    """
    Run a Python script with provided arguments using subprocess.
    Logs the output and handles errors.
    """
    cmd = [sys.executable, script_name] + args
    try:
        logging.info(f"Running {script_name} with arguments {args}")
        result = subprocess.run(cmd, check=True)
        print(f"Successfully ran {script_name}")
        logging.info(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argparse to accept all input files and necessary arguments for each script
    parser = argparse.ArgumentParser(description="Run all evaluation scripts with required inputs.")
    
    # # General input and output directories
    # parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input files.")
    # parser.add_argument('--output_dir_path', type=str, required=True, help="Directory to save output files.")
    
    # # Specific input files for each script
    # parser.add_argument('--coherence_csv', type=str, required=True, help="CSV file for coherence_vs_constraint_graph.py")
    # parser.add_argument('--model1_path', type=str, required=True, help="CSV file of Model 1 for multiple scripts (constraint_satisfaction, etc.)")
    # parser.add_argument('--model2_path', type=str, required=True, help="CSV file of Model 2 for multiple scripts (constraint_satisfaction, etc.)")
    # parser.add_argument('--model3_path', type=str, required=True, help="CSV file of Model 3 for multiple scripts (constraint_satisfaction, etc.)")
    # parser.add_argument('--label1', default="Gemma-7B Instruct", help="Label for the first model (default: Gemma-7B Instruct)")
    # parser.add_argument('--label2', default="Llama-2-7B Chat", help="Label for the second model (default: Llama-2-7B Chat)")
    # parser.add_argument('--label3', default="Mistral-7B Instruct", help="Label for the third model (default: Mistral-7B Instruct)")
    # parser.add_argument('--quc_input_json', type=str, required=True, help="JSON file for quc_and_rcs.py")
    # parser.add_argument('--story_quality_input', type=str, required=True, help="CSV file for story_quality_eval.py")

    # Model Outputs
    parser.add_argument('--model1_path', type=str, required=False, help="CSV file of Model 1 outputs for multiple scripts (constraint_satisfaction, etc.)")
    parser.add_argument('--model2_path', type=str, required=False, help="CSV file of Model 2 outputs for multiple scripts (constraint_satisfaction, etc.)")
    parser.add_argument('--model3_path', type=str, required=False, help="CSV file of Model 3 outputs for multiple scripts (constraint_satisfaction, etc.)")

    parser.add_argument('--label1', default="Gemma-7B Instruct", help="Label for the first model (default: Gemma-7B Instruct)")
    parser.add_argument('--label2', default="Llama-2-7B Chat", help="Label for the second model (default: Llama-2-7B Chat)")
    parser.add_argument('--label3', default="Mistral-7B Instruct", help="Label for the third model (default: Mistral-7B Instruct)")

    # constarint_satisfaction.py
    parser.add_argument('--input_file_path_cons_satisf', required=False, help="Path to your model's generation CSV file for constarint_satisfaction.py")
    parser.add_argument('--output_file_path_cons_satisf', required=False, help="Path to the output CSV file for constarint_satisfaction.py. New file will be generated with given name.")

    # constraint_satisfaction_graph.py
    parser.add_argument('--output_file_path_cons_satisf_graph', required=False, help="Path to save the output plot image for onstarint_satisfaction_graph.py")

    # diversity_calculation.py
    parser.add_argument('--input_path_diversity_calc', required=False, help="Path to the input CSV file for diversity_calculation.py")
    parser.add_argument('--output_path_diversity_calc', required=False, help="Path to the output CSV file where results will be saved for diversity_calculation.py")

    # diversity_graphs.py
    parser.add_argument('--output_path_diversity_graphs', required=False, help="Path to save the output plot image for diversity_graphs.py")

    # perplexity_graphs.py
    parser.add_argument('--output_path_perp_graphs', required=False, help="Path to save the output plot image for perplexity_graphs.py")

    # coherence_vs_constraint_graph.py
    parser.add_argument("--input_path_coh_vs_cons_graph", nargs='+', required=False, help="List of CSV files for each model (format: model_name file_path).")
    parser.add_argument("--output_path_coh_vs_cons_graph", required=False, help="Directory to save the output plot.")
    
    # quc_and_rcs.py
    parser.add_argument("--input_json_quc_and_rcs", required=False, help="Path to input JSON file containing grouped results.")
    parser.add_argument("--output_dir_quc_and_rcs", required=False, help="Directory to save the output plots.")
    
    # Parse arguments
    args = parser.parse_args()

    # Paths to the scripts
    script_paths = {
        "coherence_vs_constraint_graph.py": "evaluation/coherence_vs_constraint_graph.py",
        "constraint_satisfaction_graph_generation.py": "evaluation/constraint_satisfaction_graph_generation.py",
        "constraint_satisfaction.py": "evaluation/constraint_satisfaction.py",
        "diversity_calculation.py": "evaluation/diversity_calculation.py",
        "diversity_graphs.py": "evaluation/diversity_graphs.py",
        "perplexity_graphs.py": "evaluation/perplexity_graphs.py",
        "quc_and_rcs.py": "evaluation/quc_and_rcs.py",
        "story_quality_eval.py": "evaluation/story_quality_eval.py"
    }

    # Ensure all scripts exist before execution
    for script_name, script_path in script_paths.items():
        if not os.path.isfile(script_path):
            print(f"Script {script_name} not found at {script_path}")
            sys.exit(1)

    # Run each script with its respective arguments
    
    # # coherence_vs_constraint_graph.py
    # run_script(script_paths["coherence_vs_constraint_graph.py"], [
    #     '--input_csv', args.input_path_coh_vs_cons_graph, 
    #     '--output_dir', args.output_path_coh_vs_cons_graph,
    # ])
    
    # constraint_satisfaction_graph_generation.py
    run_script(script_paths["constraint_satisfaction_graph_generation.py"], [
        '--file1', args.model1_path,
        '--file2', args.model2_path,
        '--file3', args.model3_path,
        '--output_file_path', args.output_file_path_cons_satisf_graph
    ])
    
    # constraint_satisfaction.py
    # run_script(script_paths["constraint_satisfaction.py"], [
    #     '--input_path', args.input_file_path_cons_satisf,
    #     '--output_path', args.output_file_path_cons_satisf
    # ])
    
    # diversity_calculation.py
    run_script(script_paths["diversity_calculation.py"], [
        '--input_path', args.input_path_diversity_calc,
        '--output_path', args.output_path_diversity_calc
    ])
    
    # diversity_graphs.py
    run_script(script_paths["diversity_graphs.py"], [
        '--file1', args.model1_path,
        '--file2', args.model2_path,
        '--file3', args.model3_path,
        '--output_path', args.output_path_diversity_graphs
    ])
    
    # perplexity_graph_generation.py
    run_script(script_paths["perplexity_graphs.py"], [
        '--file1', args.model1_path,
        '--file2', args.model2_path,
        '--file3', args.model3_path,
        '--output_path', args.output_path_perp_graphs
    ])
    
    # # quc_and_rcs.py
    # run_script(script_paths["quc_and_rcs.py"], [
    #     '--input_json', args.input_json_quc_and_rcs,
    #     '--output_dir', args.output_dir_quc_and_rcs
    # ])
    
    # # story_quality_eval.py
    # run_script(script_paths["story_quality_eval.py"], [
    #     '--api_key', args.api_key,
    #     '--input_file', args.story_quality_input,
    #     '--output_dir', args.output_dir,
    #     '--max_trials', '35'  # Default value used here
    # ])

    print("All evaluation scripts have been successfully executed.")
    logging.info("All evaluation scripts have been successfully executed.")

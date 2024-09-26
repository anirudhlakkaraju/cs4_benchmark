# CS4: Evaluating LLM Creativity in Story Generation

## Abstract

Evaluating the creativity of large language models (LLMs) in story writing is challenging since generated stories may resemble existing narratives in the models' training data. To address this, we introduce **CS4**, a benchmark dataset with prompts of varying specificity. By increasing prompt constraints, we prevent models from reproducing known stories, indirectly assessing their creativity.

Our experiments on models like **LLaMA**, **Gemma**, and **Mistral** show the difficulty LLMs face in balancing constraint satisfaction and narrative coherence, especially with highly specific prompts. We also demonstrate that **Learning from Human Feedback (LHF)**, tested with **OLMo**, improves story selection but has limited impact on generating genuinely creative stories.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Evaluation Scripts](#evaluation-scripts)
- [Results](#results)
- [License](#license)

## Project Overview

This repository contains the code and data associated with the **CS4** benchmark, designed to evaluate the creativity of large language models (LLMs) under various levels of constraint specificity. The benchmark allows us to investigate how LLMs balance **creativity**, **constraint satisfaction**, and **coherence** in story generation tasks. Additionally, the code includes evaluation scripts for the analysis of multiple LLMs' performance on CS4.

### Key Models Evaluated:
- **LLaMA**
- **Gemma**
- **Mistral**
- **OLMo** (with insights into the impact of Learning from Human Feedback)

### Key Components:
1. **CS4 Dataset**: Contains prompts with varying degrees of specificity to test the creativity of LLMs.
2. **Evaluation Scripts**: To measure constraint satisfaction, coherence, and perplexity of the generated stories.

Here’s a streamlined version of your installation instructions that flows well and remains concise:

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/anirudhlakkaraju/llm-prompt-specificity.git
    cd llm-prompt-specificity
    ```

2. **Install Python 3.11.7 using Miniconda**:
    If you don't have Miniconda, download it from [Miniconda's website](https://docs.conda.io/en/latest/miniconda.html).

3. **Create and activate a new conda environment**:
    ```bash
    conda create --name myenv python=3.11.7
    conda activate myenv
    ```

4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Set up API keys and environment variables** for external APIs (e.g., OpenAI):

    - **Option 1: Using a `.env` file**:
        1. Create a `.env` file in your project directory.
        2. Add your API keys:
            ```plaintext
            OPENAI_API_KEY=your_openai_api_key
            OTHER_API_KEY=your_other_api_key
            ```
        3. Load the variables in your code:
            ```python
            from dotenv import load_dotenv
            load_dotenv()
            ```

    - **Option 2: Exporting directly in the terminal**:
        - For **Linux/Mac**:
            ```bash
            export OPENAI_API_KEY=your_openai_api_key
            export OTHER_API_KEY=your_other_api_key
            ```
        - For **Windows (Command Prompt)**:
            ```cmd
            set OPENAI_API_KEY=your_openai_api_key
            set OTHER_API_KEY=your_other_api_key
            ```

6. **Access the keys in your code**:
    ```python
    import os
    openai_api_key = os.getenv('OPENAI_API_KEY')
    ```

This concise format provides clear steps while maintaining a smooth flow.

## Usage

### Running the Evaluation

You can run all evaluation scripts in one step using the `run_all_evals.py` script. This script requires specific input file paths for each evaluation module. Below is an example command:

```bash
python run_all_evals.py \
    --model1_path "path/to/your/model1.csv" \
    --model2_path "path/to/your/model2.csv" \
    --model3_path "path/to/your/model3.csv" \
    --label1 "Label for Model 1" \
    --label2 "Label for Model 2" \
    --label3 "Label for Model 3" \
    --input_file_path_cons_satisf "path/to/your/input_file.csv" \
    --output_file_path_cons_satisf "path/to/your/output_file.csv" \
    --output_file_path_cons_satisf_graph "path/to/your/output_graph.png" \
    --input_path_diversity_calc "path/to/your/input_file.csv" \
    --output_path_diversity_calc "path/to/your/output_file.csv" \
    --output_path_diversity_graphs "path/to/your/output_graph.png" \
    --output_path_perp_graphs "path/to/your/output_graph.png" \
    --input_path_coh_vs_cons_graph "path/to/your/input_file.csv" \
    --output_path_coh_vs_cons_graph "path/to/your/output_file.csv" \
    --input_json_quc_and_rcs "path/to/your/input.json" \
    --output_dir_quc_and_rcs "path/to/your/output_directory"
```

### Individual Scripts

Each evaluation script can also be run independently by providing the necessary arguments. For example, to evaluate **constraint satisfaction**:

```bash
python constraint_satisfaction.py \
    --input_path /path/to/input.csv \
    --output_path /path/to/output.csv
```

## Datasets

### CS4 Benchmark Dataset

The **CS4 dataset** is designed to evaluate LLM creativity by introducing prompts with varying levels of specificity:
- **Low-Specificity Prompts**: Fewer constraints, allowing for more creative freedom.
- **High-Specificity Prompts**: Many constraints, forcing the model to produce more structured outputs.

The dataset includes multiple stories generated by LLaMA, Gemma, Mistral, and OLMo models, all with different levels of instruction complexity.

## Evaluation Scripts

### Key Metrics:
1. **Constraint Satisfaction**: Measures how well the generated story adheres to the prompt's constraints.
2. **Coherence**: Evaluates the overall coherence of the generated narrative.
3. **Diversity**: Calculates n-gram diversity to assess story originality.
4. **Perplexity**: Measures the predictability and fluency of the generated text.
5. **QUC and RCS**: Metrics specifically designed for this benchmark to evaluate creativity under constraints.

The key scripts for evaluation include:
- `coherence_vs_constraint_graph.py`: Generates graphs comparing coherence vs. constraint satisfaction.
- `constraint_satisfaction.py`: Evaluates how well generated stories satisfy constraints.
- `diversity_calculation.py`: Calculates diversity in generated stories.
- `perplexity_graph_generation.py`: Plots perplexity against the number of constraints.
- `quc_and_rcs.py`: Computes and plots QUC and RCS scores.

## Results

In our experiments:
- **LLaMA**, **Gemma**, and **Mistral** show distinct performance across varying levels of constraint specificity.
- Models struggle with maintaining creativity when the prompt becomes highly specific.
- **OLMo** demonstrates improved story selection via Learning from Human Feedback (LHF), but it struggles to generate unseen creative stories, even with LHF.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

For more information on the dataset and experiments, please refer to the accompanying research paper.
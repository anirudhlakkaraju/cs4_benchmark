import pandas as pd
import matplotlib.pyplot as plt

def plot_constraint_satisfaction():
    # Load 3 CSV files - these files have two columns (Number_of_Constraints, satisfied). 
    # The column "Number_of_Constraints" contains the number of constraints given in the prompt for story generation. 
    # The column "satisfied" contains the number (integer) of constraints that are being satisfied by the story. This has to be obtained by parsing the GPT4-constraints satisfaction results. 
    df1 = pd.read_csv("Path_to_file1.csv")
    df2 = pd.read_csv("Path_to_file2.csv")
    df3 = pd.read_csv("Path_to_file3.csv")

    # Preparing data for each dataframe - group perplexity values by the number of constraints. 
    data1 = df1.groupby('Number_of_Constraints')['satisfied'].mean()
    data2 = df2.groupby('Number_of_Constraints')['satisfied'].mean()
    data3 = df3.groupby('Number_of_Constraints')['satisfied'].mean()

    # Plotting all three datasets on the same graph
    plt.figure(figsize=(12, 6))
    data1.plot(kind='line', marker='o', label='Gemma') # Change label appropriately. 
    data2.plot(kind='line', marker='^', label='Llama') # Change label appropriately. 
    data3.plot(kind='line', marker='s', label='Mistral') # Change label appropriately. 
    plt.title('Average Percentage of Satisfaction by Number of Constraints')
    plt.xlabel('Number of Constraints')
    plt.ylabel('Average Percentage of Satisfaction')
    plt.grid(True)
    plt.xticks(list(set(data1.index).union(data2.index, data3.index)))  # Combine all x-axis values
    plt.legend()

    # Save the plot to a file
    plt.savefig('/path/to/your/directory/average_perplexity_plot.png')  # Specify your path here

    # Optionally display the plot
    plt.show()

if __name__ == "__main__":
    plot_constraint_satisfaction()

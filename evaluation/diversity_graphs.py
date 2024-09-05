import pandas as pd
import matplotlib.pyplot as plt


# The below code takes 3 CSV files as the input. These files contain two columns: Number_of_Constraints, Product_diversity. The code aggregates the diversity scores for each constraint number and draws the diversity graphs. 

def main():

    def prepare_data(df):
        # Function to calculate mean diversity grouped by 'Number_of_Constraints'
        return df.groupby('Number_of_Constraints')["Product_diversity"].mean() 

    # Load the data from CSV files
    df1 = pd.read_csv("/path/to/file1.csv")
    df2 = pd.read_csv("/path/to/file2.csv")
    df3 = pd.read_csv("/path/to/file3.csv")

    # Preparing data for each dataframe
    data1 = prepare_data(df1)
    data2 = prepare_data(df2)
    data3 = prepare_data(df3)

    # Plotting all three datasets on the same graph - modify the labels as required. 
    plt.figure(figsize=(12, 6))
    data1.plot(kind='line', marker='o', label='Gemma-7B Instruct')
    data3.plot(kind='line', marker='o', label='Mistral-7B Instruct')
    data2.plot(kind='line', marker='o', label='Llama-2-7B Chat')
    constraints = [7, 15, 23, 31, 39]

    plt.xlabel('Number of Constraints', fontsize=14)
    plt.xticks(constraints)
    plt.legend(title="Model", fontsize=12, title_fontsize='12')

    plt.ylabel('Average Diversity', fontsize=14)
    plt.grid(True)

    # Save the plot to a file specified by the user
    save_path = '/path/to/save/plot.png'
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()

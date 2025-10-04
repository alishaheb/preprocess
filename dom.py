from matplotlib import pyplot as plt
import seaborn as sns
#import dataa
from dataa import read_csv_with_encoding,df




def plot_correlation_matrix(df):
    print("Plotting correlation matrix...")
    plt.figure(figsize=(10, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

if __name__ == "__main__":

    plot_correlation_matrix(df)
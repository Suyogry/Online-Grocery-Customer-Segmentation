import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_heatmap(data, cluster_col, features):
    """Plot heatmap for cluster characteristics."""
    cluster_means = data.groupby(cluster_col)[features].mean()
    sns.heatmap(cluster_means, annot=True, cmap="coolwarm")
    plt.title("Cluster Characteristics")
    plt.savefig("results/cluster_heatmap.png")

if __name__ == "__main__":
    # Load clustered data
    data = pd.read_csv("results/cluster_summary.csv")

    # Visualize clusters
    plot_cluster_heatmap(data, cluster_col='Cluster', features=['PurchaseAmount', 'OrderFrequency', 'Age'])

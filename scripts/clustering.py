import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def perform_kmeans(data, n_clusters):
    """Perform K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data)
    return data, kmeans

if __name__ == "__main__":
    # Load preprocessed data
    data = pd.read_csv("data/processed_data.csv")

    # Perform clustering
    clustered_data, kmeans = perform_kmeans(data, n_clusters=3)

    # Save results
    clustered_data.to_csv("results/cluster_summary.csv", index=False)

    # Plot clustering result
    plt.scatter(data['PurchaseAmount'], data['OrderFrequency'], c=clustered_data['Cluster'])
    plt.xlabel('Purchase Amount')
    plt.ylabel('Order Frequency')
    plt.title('Customer Segments')
    plt.savefig("results/cluster_visualization.png")

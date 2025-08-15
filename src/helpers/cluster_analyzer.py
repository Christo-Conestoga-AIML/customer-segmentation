from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ClusterAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def find_best_k_elbow(self, k_min=2, k_max=10, plot=True):
        """
        Determines the best number of clusters using the Elbow Method.
        Only numeric columns are used.
        """
        # Keep only numeric columns
        X = self.df.select_dtypes(include=['int64', 'float64']).values

        wcss = []
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(range(k_min, k_max + 1), wcss, 'bo-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('WCSS')
            plt.title('Elbow Method for Optimal k')
            plt.show()

        for idx, val in enumerate(wcss, start=k_min):
            print(f"k={idx}: WCSS={val:.2f}")



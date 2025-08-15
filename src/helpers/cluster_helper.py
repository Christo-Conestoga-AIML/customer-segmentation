from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class KMeansClustering:
    def __init__(self, df, features=None, n_clusters=None, random_state=42):
        self.df = df.copy()
        self.features = features if features else df.select_dtypes(include='number').columns.tolist()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.X_scaled = None

    def preprocess(self):
        self.X_scaled = self.scaler.fit_transform(self.df[self.features])

    def find_optimal_k(self, max_k=10, plot=True):
        inertia = []
        for k in range(1, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(self.X_scaled)
            inertia.append(kmeans.inertia_)
        if plot:
            plt.figure(figsize=(8,5))
            plt.plot(range(1, max_k+1), inertia, marker='o')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal K')
            plt.show()
        if self.n_clusters is None:
            self.n_clusters = int(input("Enter the chosen number of clusters based on the elbow plot: "))
        return self.n_clusters

    def fit(self):
        if self.X_scaled is None:
            self.preprocess()
        if self.n_clusters is None:
            raise ValueError("Number of clusters not defined. Pass n_clusters to the class or use find_optimal_k().")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)
        return self.df

    def visualize_clusters(self):
        if self.X_scaled is None:
            self.preprocess()
        if 'Cluster' not in self.df.columns:
            raise ValueError("Clusters not computed. Call fit() first.")
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.X_scaled)
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=components[:,0], y=components[:,1], hue=self.df['Cluster'], palette='Set2')
        plt.title('Customer Segments')
        plt.show()

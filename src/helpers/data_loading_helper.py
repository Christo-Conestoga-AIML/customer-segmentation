import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
            self.df = pd.read_csv(self.file_path, nrows=10000)
            return self.df


    def plot_event_distribution(self):
        sns.countplot(x='event_type', data=self.df)
        plt.title('Event Type Distribution')
        plt.show()


    def plot_products_with_type(self, type_name, x_label):
        top_purchased_products = self.df[self.df['event_type'] == type_name]['product_id'].value_counts().head(10)
        top_purchased_products.plot(kind='barh', figsize=(8, 5))
        plt.title(f'Top 10 {x_label} Products')
        plt.xlabel(f'Number of time item is {x_label}')
        plt.ylabel('Product ID')
        plt.show()


    def check_missing_values(self):
        missing = self.df.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing Values']
        return missing

    def drop_missing_values(self):
        before_shape = self.df.shape
        self.df.dropna(inplace=True)
        after_shape = self.df.shape
        print(f"Dropped {before_shape[0] - after_shape[0]} rows containing missing values.")
        print(f"New dataset shape: {after_shape}")

    def check_duplicates(self, drop=False):
        dup_count = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {dup_count}")

        if drop and dup_count > 0:
            before_shape = self.df.shape
            self.df.drop_duplicates(inplace=True)
            after_shape = self.df.shape
            print(f"Dropped {before_shape[0] - after_shape[0]} duplicate rows.")
            print(f"New dataset shape: {after_shape}")

        return dup_count

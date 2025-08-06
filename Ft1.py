# plot_color_distribution.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_color_dist(csv_file):
    df = pd.read_csv(csv_file)
    color_counts = df['primary_color'].value_counts()
    color_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Primary Color')
    plt.ylabel('Number of Butterflies')
    plt.title('Butterflies by Dominant Color (CV-Classified)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file = "color_results.csv"
    plot_color_dist(csv_file)

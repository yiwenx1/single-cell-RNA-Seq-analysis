import numpy as np
import pandas as pd
import scprep
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import dimension_reduction

def plot_heatmap(filename):
    n_clusters = 9
    matrix = pd.read_csv(filename)
    print(matrix.head())
    matrix = matrix.to_numpy()
    matrix = matrix[1:, 1:]
    matrix = np.array(matrix, dtype=float)
    print(np.max(matrix))
    # v = np.min(matrix)
    # myData = [{"points": x-v, "label": np.random.randint(n_clusters)} for x in matrix]
    # plt.scatter([actdata['points'][0] for actdata in myData], [actdata['points'][1] for actdata in myData], c=[actdata['label'] for actdata in myData], s=7)
    # plt.show()
    sns.heatmap(matrix)
    plt.show()

def data(filename):
    matrix = pd.read_csv(filename)
    matrix = matrix.to_numpy()
    matrix = matrix[1:, 1:]
    matrix = np.array(matrix, dtype=float)
    pca_data, singular_values = scprep.reduce.pca(matrix, n_components=100, return_singular_values=True)
    scprep.plot.scree_plot(singular_values)
    plt.show()
    scprep.plot.scree_plot(singular_values, cumulative=True)
    plt.show()

def scatter(filename):
    matrix = pd.read_csv(filename)
    matrix = matrix.to_numpy()
    matrix = matrix[1:, 1:]
    matrix = np.array(matrix, dtype=float)
    colors = matrix[:, 0]
    scprep.plot.scatter2d(matrix, colors)
    plt.show()

def expression_level(filename):
    matrix = pd.read_csv(filename)
    matrix = matrix.to_numpy()
    matrix = matrix[1:, 1:]
    matrix = np.array(matrix, dtype=float)
    pca_data, singular_values = scprep.reduce.pca(matrix, n_components=100, return_singular_values=True)
    scprep.plot.scatter3d(pca_data)
    plt.show()

if __name__ == "__main__":
    plot_heatmap("./data/normalized_matrix.csv")
    plot_heatmap("./data/magic.csv")
    # expression_level("./data/normalized_matrix.csv")
    # expression_level("./data/magic.csv")
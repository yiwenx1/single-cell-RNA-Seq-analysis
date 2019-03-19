import numpy as np
import pandas as pd
import scanpy as sc

import magic
import scprep

import numpy as np
import matplotlib.pyplot as plt

def filtering(matrix):
    filtered = scprep.filter.filter_empty_cells(matrix)
    filtered = scprep.filter.filter_empty_genes(filtered)
    filtered = scprep.filter.filter_gene_set_expression(filtered, percentile=30)

    print(filtered.shape)
    return filtered

def normalization(matrix):
    normalized = scprep.normalize.batch_mean_center(matrix)
    normalized = scprep.normalize.library_size_normalize(normalized)
    print(normalized.shape)
    return normalized

def dimension_reduction(matrix, algorithm):
    c = None
    if algorithm == "SVD":
        c = scprep.reduce.AutomaticDimensionSVD()
    elif algorithm == "Random":
        c = scprep.reduce.InvertibleRandomProjection()
    else:
        c = scprep.reduce.SparseInputPCA()
    reduced_matrix = c.fit_transform(matrix)
    print(reduced_matrix.shape)
    return reduced_matrix
        

if __name__ == "__main__":
    matrix = np.load('data/7_matrix.npy')
    filtered_matrix = filtering(matrix)
    normalized_matrix = normalization(filtered_matrix)
    algorithm = "Random"
    reduced_matrix = dimension_reduction(normalized_matrix, algorithm)
    # np.save('./data/processed_matrix.npy', reduced_matrix)
    np.savetxt('./data/processed_matrix.csv', reduced_matrix, delimiter=',')
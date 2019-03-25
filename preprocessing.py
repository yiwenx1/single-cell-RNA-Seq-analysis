import numpy as np
import magic
import scprep
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import resample

def add_name(matrix, cellfile, genefile):
    cell_names = []
    count = 0
    with open(cellfile) as f:
        for line in f:
            line = line.strip().split(",")
            count += 1
            if count > 1:
                cell_names.append(line[1])
    
    gene_names = []
    count = 0
    with open(genefile) as f:
        for line in f:
            line = line.strip().split(",")
            count += 1
            if count > 1:
                gene_names.append(line[1])
    
    return pd.DataFrame(matrix, index=cell_names, columns=gene_names)

def filtering(matrix):
    print("Aftering loading:", matrix.shape, sum(matrix[matrix==0].count(axis=1))/ sum(matrix.count()))
    # downsampling
    matrix = resample(matrix, replace=False, n_samples = 5000, random_state=35)
    print("downsampleing")
    # scprep.plot.plot_library_size(matrix, cutoff=1000)
    # plt.show()
    print("Aftering downsampling:", matrix.shape, sum(matrix[matrix==0].count(axis=1))/ sum(matrix.count()))

    filtered = scprep.filter.filter_empty_cells(matrix)
    filtered = scprep.filter.filter_empty_genes(filtered)
    filtered = scprep.filter.filter_library_size(filtered, cutoff=1000)
    filtered = scprep.filter.filter_rare_genes(filtered, cutoff=150)

    print("Aftering filtering:", filtered.shape, sum(filtered[filtered==0].count(axis=1))/ sum(filtered.count()))
    return filtered

def normalization(matrix):
    normalized = scprep.normalize.library_size_normalize(matrix)
    normalized = scprep.transform.sqrt(normalized)

    print("After normalization:", normalized.shape, sum(normalized[normalized==0].count(axis=1))/ sum(normalized.count()))
    print(normalized.head())
    return normalized

def magic_process(matrix):
    magic_op = magic.MAGIC(knn=10)
    magiced = magic_op.fit_transform(matrix, genes="all_genes")

    print("after MAGIC:", magiced.shape, sum(magiced[magiced==0].count(axis=1)) / sum(magiced.count()))
    print(magiced.head())
    return magiced, magic_op

def gene_gene_relationships(matrix, matrix_magic):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    scprep.plot.scatter(x=matrix['ENSG00000186842_LINC00846'], y=matrix['ENSG00000159128_IFNGR2'], 
                        c=matrix['ENSG00000244676_AL109761.5'], ax=ax1,
                        xlabel='ENSG00000186842_LINC00846', ylabel='ENSG00000159128_IFNGR2', 
                        legend_title='ENSG00000244676_AL109761.5', title="Brfore MAGIC")
    scprep.plot.scatter(x=matrix_magic['ENSG00000186842_LINC00846'], y=matrix_magic['ENSG00000159128_IFNGR2'], 
                        c=matrix_magic['ENSG00000244676_AL109761.5'], ax=ax2,
                        xlabel='ENSG00000186842_LINC00846', ylabel='ENSG00000159128_IFNGR2', 
                        legend_title='ENSG00000244676_AL109761.5', title="After MAGIC")
    plt.tight_layout()
    plt.show()

def cell_trajectory(matrix, matrix_magic, magic_op):
    magic_pca = magic_op.transform(genes="pca_only")
    matrix_pca = PCA(n_components=3).fit_transform(np.array(matrix))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    scprep.plot.scatter2d(matrix_pca, c=matrix['ENSG00000100429_HDAC10'],
                          label_prefix='PC', title='PCA without MAGIC',
                          legend_title='ENSG00000100429_HDAC10', ax=ax1, ticks=False)
    scprep.plot.scatter2d(magic_pca, c=matrix_magic['ENSG00000100429_HDAC10'],
                          label_prefix='PC', title='PCA with MAGIC',
                          legend_title='ENSG00000100429_HDAC10', ax=ax2, ticks=False)
    plt.tight_layout()
    plt.show()


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
    matrix = add_name(matrix, "./data/GSM2396858_k562_tfs_7_cellnames.csv", "./data/GSM2396858_k562_tfs_7_genenames.csv")
    
    # Plot the library size histogram of raw data.
    # print("raw data")
    # scprep.plot.plot_library_size(matrix)
    # plt.show()

    filtered_matrix = filtering(matrix)
    normalized_matrix = normalization(filtered_matrix)
    # np.savetxt('./data/normalized_matrix.csv', normalized_matrix, delimiter=',')
    # normalized_matrix.to_csv('./data/normalized_matrix.csv')
    
    magiced_matrix, magic_op = magic_process(normalized_matrix)
    # magiced_matrix_transpose = magiced_matrix.T
    # magiced_matrix.to_csv('./data/magic.csv')
    # magiced_matrix_transpose.to_csv('./data/magic_T.csv')

    # gene gene relationships.
    # gene_gene_relationships(normalized_matrix, magiced_matrix)

    # Visualizing cell trajectories with PCA
    # cell_trajectory(normalized_matrix, magiced_matrix, magic_op)

    # algorithm = "PCA"
    # reduced_matrix = dimension_reduction(normalized_matrix, algorithm)
    # np.save('./data/PCA_matrix.npy', reduced_matrix)
    # np.savetxt('./data/PCA_matrix.csv', reduced_matrix, delimiter=',')
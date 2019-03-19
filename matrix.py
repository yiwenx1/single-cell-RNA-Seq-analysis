import numpy as np
import scipy.io
"""
k562_tfs_7
n_genes = 23111 (id: 0-23110)
n_cells = 33013 (id: 0-33012)
"""

def convert2matrix(filename, n_genes, n_cells, output_filename):
    matrix = np.zeros((n_cells, n_genes))
    count = 0
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            count += 1
            if count > 3:
                cell = int(line[0])
                gene = int(line[1])
                umi_count = int(line[2])
                if cell < n_cells and gene < n_genes:
                    matrix[cell][gene] = umi_count
    np.save(output_filename, matrix)

if __name__ == "__main__":
    filename = "data/GSM2396858_k562_tfs_7.mtx.txt"
    n_cells = 33013
    n_genes = 23111
    convert2matrix(filename, n_genes, n_cells, "data/7_matrix.mtx")
import numpy as np


def p_val_matrix_to_vec(matrix):

    ret_vals = []
    for r in range(matrix.shape[0]):
        for c in range(r + 1,matrix.shape[1]):
            ret_vals.append(matrix[r,c])

    return ret_vals

def p_val_vec_to_matrix(p_vec, num_rows,symmetrize=False):
    p_val_matrix = np.zeros( (num_rows, num_rows) )

    cnt = 0

    for r in range(p_val_matrix.shape[0]):
        for c in range(r,p_val_matrix.shape[1]):
            if r == c:
                p_val_matrix[r,c] = 1
                continue
            
            p_val_matrix[r,c] = p_vec[cnt]
            if symmetrize:
                p_val_matrix[c,r] = p_vec[cnt]
            cnt +=1

    return p_val_matrix

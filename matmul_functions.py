import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    num_rows = len(X)
    num_cols = len(X[0])
    result = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i + 1):
            for k in range(num_cols):
                result[i][j] += X[i][k] * X[j][k]
            result[j][i] = result[i][j]

    return result

@njit(parallel=True)
def matmul_transpose_numba(X):
    num_rows = len(X)
    num_cols = len(X[0])
    result = np.zeros((num_rows, num_rows))
    for i in prange(num_rows):
        for j in prange(i + 1):
            for k in prange(num_cols):
                result[i][j] += X[i][k] * X[j][k]
            result[j][i] = result[i][j]

    return result


def matmul_transpose_gpu(X):
    length = len(X)
    C = np.zeros((length, length))
    threadsperblock = 1024
    blockspecrgrid = 1
    d_X = cuda.to_device(X)
    d_C = cuda.to_device(C)
    matmul_kernel[blockspecrgrid, threadsperblock](d_X, d_C)
    result_array = d_C.copy_to_host()
    return result_array


@cuda.jit
def matmul_kernel(X, C):
    num_rows = len(X)
    num_cols = len(X[0])
    tx = cuda.threadIdx.x
    cell_number = int(tx) # We number each cell, line by line, from left to right, from 0 to (length * length - 1)

    while cell_number < num_rows * num_rows:
        curr_row = cell_number // num_rows
        curr_col = cell_number % num_rows

        if curr_row >= curr_col: # We only calculate the cells in and under the main diagonal
            for k in prange(num_cols):
                C[curr_row][curr_col] += X[curr_row][k] * X[curr_col][k]
            C[curr_col][curr_row] = C[curr_row][curr_col] # The cells above the main diagonal are filled here

        cell_number += 1024 # We use 1024 threads, the thread tx calculate the cells tx + 1024 * i where i = 0,1,2...


#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()

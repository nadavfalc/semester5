import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def dist_cpu(A, B, p):
    sum = 0
    for i in range(1000):
        for j in range(1000):
            sum += (abs(A[i][j] - B[i][j])) ** p

    result = sum ** (1.0/p)
    return np.array([result])


@njit(parallel=True)
# Same as dist_cpu, but here we use prange instead of range
def dist_numba(A, B, p):
    sum = 0
    for i in prange(1000):
        for j in prange(1000):
            sum += (abs(A[i][j] - B[i][j])) ** p

    result = sum ** (1.0/p)
    return np.array([result])


def dist_gpu(A, B, p):
    C = np.zeros(1)
    threadsperblock = 1000
    blockspergrid = 1000
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    dist_kernel[blockspergrid, threadsperblock](d_A,d_B,p,C)
    return np.power(C , 1.0 / p)


@cuda.jit
def dist_kernel(A, B, p, C):
    tx = cuda.threadIdx.x
    tb = cuda.blockIdx.x
    val = (abs(A[tx][tb] - B[tx][tb])) ** p
    cuda.atomic.add(C,0,val)


#this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0,256,(1000, 1000))
    B = np.random.randint(0,256,(1000, 1000))
    p = [1, 2]

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))


    for power in p:
        print('p=' + str(power))
        print('     [*] CPU:', timer(dist_cpu,power))
        print('     [*] Numba:', timer(dist_numba,power))
        print('     [*] CUDA:', timer(dist_gpu, power))


if __name__ == '__main__':
    dist_comparison()

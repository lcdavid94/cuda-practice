#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include "common.h"

#ifndef SILENT
#define PRINT(...) printf(__VA_ARGS__)
#else
#define PRINT(...)
#endif

const int matDim1 = 1024;
const int matDim2 = 1024;
const int matDim3 = 1024;
const int maxGridSize = 64;
const int maxBlockSize = 64;

__global__ void matMult(int *a, int *b, int *c) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    while (idx < matDim1 * matDim3) {
        temp[tid] = 0;
        int x = idx / matDim3;
        int y = idx % matDim3;
        int eid = tid;
        while (eid < matDim2) {
            temp[tid] += a[x*matDim2+eid]*b[eid*matDim3+y];
            eid += blockDim.x;
        }
        __syncthreads();
        int tot = blockDim.x;
        while (tot > 1) {
            int size = (tot+1) / 2;
            if (tid + size < tot) {
                temp[tid] += temp[tid+size];
            }
            __syncthreads();
            tot = size;
        }
        c[idx] = temp[0];
        idx += gridDim.x;
    }
}

void printMat(int *mat, int r, int c) {
    for (int i=0; i<r; ++i) {
        for (int j=0; j<c; ++j) {
            PRINT("%4d", mat[i*c+j]);
        }
        PRINT("\n");
    }
}

int main() {
    srand((int)time(0));
    int *a, *b, *c;
    HANDLE_ERROR(cudaMallocManaged(&a, sizeof(int[matDim1*matDim2])));
    HANDLE_ERROR(cudaMallocManaged(&b, sizeof(int[matDim2*matDim3])));
    HANDLE_ERROR(cudaMallocManaged(&c, sizeof(int[matDim1*matDim3])));
    for (int i=0; i<matDim1; ++i) {
        for (int j=0; j<matDim2; ++j) {
            a[i*matDim2+j] = rand() % 5;
        }
    }
    for (int i=0; i<matDim2; ++i) {
        for (int j=0; j<matDim3; ++j) {
            b[i*matDim3+j] = rand() % 5;
        }
    }
    int gridSize = min(matDim1*matDim3, maxGridSize);
    int blockSize = min(matDim2, maxBlockSize);
    matMult<<<gridSize, blockSize, blockSize>>>(a, b, c);
    HANDLE_ERROR(cudaDeviceSynchronize());
    PRINT("\na =\n");
    printMat(a, matDim1, matDim2);
    PRINT("\nb =\n");
    printMat(b, matDim2, matDim3);
    PRINT("\nc =\n");
    printMat(c, matDim1, matDim3);
#ifdef CHECK
    printf("Checking...\n");
    for (int i=0; i<matDim1; ++i) {
        for (int j=0; j<matDim3; ++j) {
            int ans = 0;
            for (int k=0; k<matDim2; ++k) {
                ans += a[i*matDim2+k] * b[k*matDim3+j];
            }
            if (ans != c[i*matDim3+j]) {
                printf("%d %d should be %d\n", i, j, ans);
            }
        }
    }
#endif
    HANDLE_ERROR(cudaFree(a));
    HANDLE_ERROR(cudaFree(b));
    HANDLE_ERROR(cudaFree(c));
}

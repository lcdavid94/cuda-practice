#include <stdio.h>
#include <unistd.h>
#define HANDLE_ERROR(x) {\
    cudaError_t status = x;\
    if (status) {\
        printf("Error %d line %d\n", status, __LINE__);\
    }\
}

const int chunk = 10;
const int limit = 100;
const int target = 2000;

#define check_and_add(M) {\
    mult(X, M, Y);\
    if (!dup(&queue, Y)) {\
        addToArray(&queue, Y);\
    }\
}

struct array {
    int **data;
    int length;
    int cap;
};

__device__ void initArray(array *arr) {
    arr->cap = 4;
    arr->data = (int**) malloc(sizeof(int*)*arr->cap);
    arr->length = 0;
}

__device__ void deinitArray(array *arr) {
    if (arr) {
        for (int i=0; i<arr->length; ++i) {
            free(arr->data[i]);
        }
        free(arr->data);
    }
}

__device__ void addToArray(array *arr, int *add) {
    if (arr->length == arr->cap) {
        int newcap = arr->cap * 2;
        int **newdata = (int**) malloc(sizeof(int*)*newcap);
        memcpy(newdata, arr->data, sizeof(int*)*arr->cap);
        arr->cap = newcap;
        arr->data = newdata;
    }
    int *newcpy = (int*) malloc(sizeof(int)*25);
    memcpy(newcpy, add, sizeof(int)*25);
    arr->data[arr->length++] = newcpy;
}

__device__ bool same(int *a, int *b) {
    for (int i=0; i<25; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__device__ bool dup(array *queue, int *arr) {
    for (int i=0; i<queue->length; ++i) {
        if (same(queue->data[i], arr)) {
            return true;
        }
    }
    return false;
}

__device__ void mult(int *a, int *b, int *c) {
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            c[i*5+j] = 0;
            for (int k=0; k<5; ++k) {
                c[i*5+j] += a[i*5+k]*b[k*5+j];
            }
        }
    }
}

__global__ void find(int a, int b, int *ans) {
    int baseA = a+blockIdx.x, baseB = b+threadIdx.x;
    if (baseA < limit && baseB < limit) {
        int A[25], B[25];
        for (int i=0; i<25; ++i) {
            A[i] = baseA % 3 - 1;
            B[i] = baseB % 3 - 1;
            baseA /= 3;
            baseB /= 3;
        }
        array queue;
        initArray(&queue);
        addToArray(&queue, A);
        addToArray(&queue, B);
        int idx = 0;
        while (idx < queue.length && queue.length < target) {
            int *X = queue.data[idx];
            int Y[25];
            check_and_add(A);
            check_and_add(B);
            ++idx;
        }
        if (idx == queue.length && queue.length > *ans) {
            *ans = queue.length;
        }
        deinitArray(&queue);
    }

}

int main() {
    int *p;
    HANDLE_ERROR(cudaMallocManaged(&p, sizeof(int)));
    for (int i = 0; i<limit; i+= chunk) {
        for (int j = 0; j<limit; j+= chunk) {
            find<<<chunk, chunk>>>(i, j, p);
            cudaError_t status = cudaDeviceSynchronize();
            printf("status = %d\n", status);
            printf("ans = %d\n", *p);
            sleep(1);
        }
    }
}

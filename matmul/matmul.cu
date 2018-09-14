#include <cstdio>
#include <cstdlib>
const int DIM = 512;
const int N = DIM*DIM;
const int CUDA_DIM = 64;
const int MOD = 5;

void printMat(int *m) {
#ifndef NO_PRINT
    for (int i=0; i<N; ++i) {
        if (i%DIM==DIM-1) {
            printf("\n");
        }
    }
#endif
}

struct mat {
    int *p;
    mat(int n) {
        int ret = cudaMalloc(&p, sizeof(int[DIM*n]));
        if (ret) {
            printf("Error code: %d\n", ret);
        }
    }
    __device__ int& operator[](int n) {
        return p[n];
    }
    int* operator+(int n) {
        return p+n;
    }
    ~mat() {
        cudaFree(p);
    }
};

__global__ void matAddOrSub(int *a, int *b, int *c, bool add) {
    int x = blockIdx.x, y = threadIdx.x;
    int offset = x*DIM + y;
    c[offset] = add ? a[offset]+b[offset]:a[offset]-b[offset];
}

__global__ void matMulCuda(int *a, int *b, int *c) {
    int x = blockIdx.x, y = threadIdx.x;
    int offset = x*DIM + y;
    int dim = gridDim.x;
    c[offset] = 0;
    for (int i=0; i<dim; ++i)
        c[offset] += a[x*DIM+i]*b[i*DIM+y];
}

void matMul(int *a, int *b, int *c, int dim) {
    if (dim <= CUDA_DIM) {
        matMulCuda<<<dim, dim>>>(a, b, c);
        return;
    }
    mat p1(dim), p2(dim), p3(dim);
    int half = dim/2;
    int *a11=a, *a12=a+half, *a21=a+half*DIM, *a22=a+half*DIM+half;
    int *b11=b, *b12=b+half, *b21=b+half*DIM, *b22=b+half*DIM+half;
    int *s1=p1+0, *s2=p1+half, *s3=p1+half*DIM, *s4=p1+half*DIM+half;
    int *s5=p2+0, *s6=p2+half, *s7=p2+half*DIM, *s8=p2+half*DIM+half;
    int *s9=p3+0, *s10=p3+half;
    matAddOrSub<<<half, half>>>(a11, a22, s1, true);
    matAddOrSub<<<half, half>>>(b11, b22, s2, true);
    matAddOrSub<<<half, half>>>(a21, a22, s3, true);
    matAddOrSub<<<half, half>>>(b12, b22, s4, false);
    matAddOrSub<<<half, half>>>(b21, b11, s5, false);
    matAddOrSub<<<half, half>>>(a11, a12, s6, true);
    matAddOrSub<<<half, half>>>(a21, a11, s7, false);
    matAddOrSub<<<half, half>>>(b11, b12, s8, true);
    matAddOrSub<<<half, half>>>(a12, a22, s9, false);
    matAddOrSub<<<half, half>>>(b21, b22, s10, true);
    mat q1(dim), q2(dim);
    int *m1=q1+0, *m2=q1+half, *m3=q1+half*DIM, *m4=q1+half*DIM+half;
    int *m5=q2+0, *m6=q2+half, *m7=q2+half*DIM;
    matMul(s1, s2, m1, half);
    matMul(s3, b11, m2, half);
    matMul(a11, s4, m3, half);
    matMul(a22, s5, m4, half);
    matMul(s6, b22, m5, half);
    matMul(s7, s8, m6, half);
    matMul(s9, s10, m7, half);
    int *c11=c, *c12=c+half, *c21=c+half*DIM, *c22=c+half*DIM+half;
    matAddOrSub<<<half, half>>>(m1, m4, c11, true);
    matAddOrSub<<<half, half>>>(c11, m5, c11, false);
    matAddOrSub<<<half, half>>>(c11, m7, c11, true);
    matAddOrSub<<<half, half>>>(m3, m5, c12, true);
    matAddOrSub<<<half, half>>>(m2, m4, c21, true);
    matAddOrSub<<<half, half>>>(m1, m2, c22, false);
    matAddOrSub<<<half, half>>>(c22, m3, c22, true);
    matAddOrSub<<<half, half>>>(c22, m6, c22, true);
}

int main() {
    int a[N], b[N], c[N];
    for (int i=0; i<N; ++i) {
        a[i] = rand()%MOD;
        b[i] = rand()%MOD;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int *devA, *devB, *devC;
    cudaMalloc(&devA, sizeof(int[N]));
    cudaMalloc(&devB, sizeof(int[N]));
    cudaMalloc(&devC, sizeof(int[N]));
    cudaMemcpy(devA, a, sizeof(int[N]), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, sizeof(int[N]), cudaMemcpyHostToDevice);
    matMul(devA, devB, devC, DIM);
    cudaMemcpy(c, devC, sizeof(int[N]), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    printf("a:\n");
    printMat(a);
    printf("\nb:\n");
    printMat(b);
    printf("\nc:\n");
    printMat(c);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nTime taken for matrix multiplication: %.3f ms", elapsedTime);

    int ans[N];
    for (int i=0; i<DIM; ++i) {
        for (int j=0; j<DIM; ++j) {
            int offset = i*DIM+j;
            ans[offset] = 0;
            for (int k=0; k<DIM; ++k) {
                ans[offset] += a[i*DIM+k]*b[k*DIM+j];
            }
        }
    }
    try {
        for (int i=0; i<N; ++i) {
            if (ans[i]!=c[i]) {
                throw 1;
            }
        }
        printf("\nSuccess!\n");
    } catch (int) {
        printf("\nFailed\n");
    }

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

#pragma once
#include <stdio.h>
#define HANDLE_ERROR(exp) do {\
    cudaError_t status = exp;\
    if (status) {\
        printf("Error %d: %s\n", status, cudaGetErrorString(status));\
        exit(1);\
    }\
} while(0)

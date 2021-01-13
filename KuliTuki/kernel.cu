#include <cstdlib>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

struct complex {
    double real;
    double imaginary;
};


#define N 16384
#define PI 3.14159265359


__device__
complex multiplyComplex(complex a, complex b)
{
    complex result;
    result.real = a.real * b.real - a.imaginary * b.imaginary;
    result.imaginary = a.real * b.imaginary + b.real * a.imaginary;

    return result;
}

__device__
complex addComplex(complex a, complex b)
{
    complex result = { a.real + b.real, a.imaginary + b.imaginary };
    return result;
}

__device__
complex subtractComplex(complex a, complex b)
{
    complex result = { a.real - b.real, a.imaginary - b.imaginary };
    return result;
}

__global__
void compute_Even_Odd(complex* xInput_d, complex* even_d, complex* odd_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx / (N / 2);
    int m = idx % (N / 2);

    // считаем экспоненту
    double exp = -4.0 * PI * m * k / N;

    complex e_part;

    // считаем часть с e^(i * exponent)
    e_part.real = cos(exp);
    e_part.imaginary = sin(exp);

    // сохраняем чётную часть
    even_d[idx] = multiplyComplex(xInput_d[2 * m], e_part);

    // сохраняем нечётную часть
    odd_d[idx] = multiplyComplex(xInput_d[2 * m + 1], e_part);
}


// вычисляет twiddle коэффициент для заданных значений N и k.
__device__
void compute_twiddle(complex* twiddle, int* k)
{
    double exp = -2.0 * PI * (*k) / N;
    twiddle->real = cos(exp);
    twiddle->imaginary = sin(exp);
}


__global__
void sum_Regression(complex* even_d, complex* odd_d, complex* result_d)
{
    int n = N / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx / (n / 2);
    int m = idx % (n / 2);
    int z = k * n + m;

    while (n > 1)
    {
        __syncthreads();
        if (m < n / 2)
        {
            even_d[z] = addComplex(even_d[z], even_d[z + n / 2]);
            odd_d[z] = addComplex(odd_d[z], odd_d[z + n / 2]);
        }
        n = n / 2;
    }
    if (m == 0)
    {
        compute_twiddle(result_d + k, &k);
        // считаем twiddle * O_k
        odd_d[z] = multiplyComplex(odd_d[z], result_d[k]);

        // считаем X_k и X_k + N/2
        result_d[k] = addComplex(even_d[z], odd_d[z]);
        result_d[k + N / 2] = subtractComplex(even_d[z], odd_d[z]);
    }
}


int main() {
    complex* xInput, * result;

    complex* xInput_d, * result_d;
    complex* even_d, * odd_d;


    // выделяем память
    xInput = (complex*)malloc(N * sizeof(complex));
    result = (complex*)malloc(N * sizeof(complex));

    cudaMalloc((void**)&xInput_d, N * sizeof(complex));
    cudaMalloc((void**)&result_d, N * sizeof(complex));

    // установим определенные значения от 0 до 7 для xInput
    /*xInput[0].real = 3.6;
    xInput[0].imaginary = 2.6;
    xInput[1].real = 2.9;
    xInput[1].imaginary = 6.3;
    xInput[2].real = 5.6;
    xInput[2].imaginary = 4;
    xInput[3].real = 4.8;
    xInput[3].imaginary = 9.1;
    xInput[4].real = 3.3;
    xInput[4].imaginary = 0.4;
    xInput[5].real = 5.9;
    xInput[5].imaginary = 4.8;
    xInput[6].real = 5;
    xInput[6].imaginary = 2.6;
    xInput[7].real = 4.3;
    xInput[7].imaginary = 4.1;*/

    for (int i = 0; i < N ; i++)
    {
        xInput[i].real = 1.0f;
        xInput[i].imaginary = 0.0f;
    }

    clock_t st = clock();
    cudaMemcpy(xInput_d, xInput, sizeof(complex) * N, cudaMemcpyHostToDevice);

    int m_blocks, m_threads;
    m_threads = (N / 2) * (N / 2); // k переходит от 0 к N - 1,
                                   // m переходит от 0 к N/2 - 1 для каждого значения k

    cudaMalloc((void**)&even_d, sizeof(complex) * m_threads);
    cudaMalloc((void**)&odd_d, sizeof(complex) * m_threads);

    m_blocks = N / 1024; // количество необходимых нам блоков

    if (m_blocks == 0)
        m_blocks = 1;
    else
        m_threads = 1024;

    // Используем половину блоков для вычисления результатов с помощью регрессии
    dim3 dimSumGrid(m_blocks / 2, 1);
    dim3 dimSumBlock(m_threads, 1);


    if (m_blocks == 1) {
        dimSumGrid = dim3(m_blocks, 1);
        dimSumBlock = dim3(m_threads / 2, 1);
    }


    cudaEvent_t start, stop;
    float elapsedTime;


    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    compute_Even_Odd <<<m_blocks, m_threads>>> (xInput_d, even_d, odd_d);

    sum_Regression <<<dimSumGrid, dimSumBlock>>> (even_d, odd_d, result_d);

    clock_t end = clock();
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Cuda time : %f milli secs\n", elapsedTime);

    cudaMemcpy(result, result_d, sizeof(complex) * N, cudaMemcpyDeviceToHost);

    cudaFree(xInput_d);
    cudaFree(result_d);
    cudaFree(even_d);
    cudaFree(odd_d);

    printf("PARALLEL VERSION\n\n");
    printf("TOTAL PROCESSED SAMPLES: %d\n", N);

    for (int i = 0; i < 8; i++)
    {
        printf("==================\n");
        printf("XR[%d]: %f\n", i, result[i].real);
        printf("XI[%d]: %f\n", i, result[i].imaginary);
    }

    printf("==================\n");

    printf("\nC clock Time: %f\n", ((double)(end - st)) / CLOCKS_PER_SEC);


    return 0;
}

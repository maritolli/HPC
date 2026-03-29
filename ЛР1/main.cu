#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

// прроверка ошибок CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// CPU реализация
void multiplyCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU реализация
__global__ void multiplyGPUKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void multiplyGPU(const float* d_A, const float* d_B, float* d_C, int N, dim3 blockSize) {
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
        (N + blockSize.y - 1) / blockSize.y);
    multiplyGPUKernel <<<gridSize, blockSize>>> (d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
}


// ф-ия проверки результатов
bool verifyResults(const float* cpu, const float* gpu, int N, float epsilon = 1e-2f) {
    for (int i = 0; i < N * N; i++) {
        if (std::abs(cpu[i] - gpu[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void initMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand() % 10);
    }
}

int main() {
    const int STEP = 100;
    const int MAX_SIZE = 2000;
    const dim3 BLOCK_SIZE(16, 16);

    std::cout << std::setw(10) << "Size"
        << std::setw(15) << "CPU Time (ms)"
        << std::setw(15) << "GPU Time (ms)"
        << std::setw(15) << "Speedup"
        << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int N = STEP; N <= MAX_SIZE; N += STEP) {
        int sizeBytes = N * N * sizeof(float);

        // Host Memory
        std::vector<float> h_A(N * N), h_B(N * N), h_C_cpu(N * N), h_C_gpu(N * N);
        initMatrix(h_A.data(), N);
        initMatrix(h_B.data(), N);

        // Device Memory
        float* d_A, * d_B, * d_C;
        cudaCheckError(cudaMalloc(&d_A, sizeBytes));
        cudaCheckError(cudaMalloc(&d_B, sizeBytes));
        cudaCheckError(cudaMalloc(&d_C, sizeBytes));

        // Copy to Device
        cudaCheckError(cudaMemcpy(d_A, h_A.data(), sizeBytes, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_B, h_B.data(), sizeBytes, cudaMemcpyHostToDevice));

        // время выполнения на CPU
        auto start_cpu = std::chrono::high_resolution_clock::now();
        multiplyCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> time_cpu = end_cpu - start_cpu;

        // время выполнения на GPU Timing
        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&stop));

        cudaCheckError(cudaEventRecord(start));
        multiplyGPU(d_A, d_B, d_C, N, BLOCK_SIZE);
        cudaCheckError(cudaEventRecord(stop));
        cudaCheckError(cudaEventSynchronize(stop));

        float time_gpu = 0;
        cudaCheckError(cudaEventElapsedTime(&time_gpu, start, stop));

        cudaCheckError(cudaMemcpy(h_C_gpu.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost));

        bool ok = verifyResults(h_C_cpu.data(), h_C_gpu.data(), N);
        float speedup = (time_gpu > 0) ? time_cpu.count() / time_gpu : 0;

        std::cout << std::setw(10) << N << "x" << N
            << std::setw(15) << std::fixed << std::setprecision(2) << time_cpu.count()
            << std::setw(15) << time_gpu
            << std::setw(15) << std::setprecision(2) << speedup
            << std::setw(15) << (ok ? "CORRECT" : "FAIL") << std::endl;

        cudaCheckError(cudaFree(d_A));
        cudaCheckError(cudaFree(d_B));
        cudaCheckError(cudaFree(d_C));
        cudaCheckError(cudaEventDestroy(start));
        cudaCheckError(cudaEventDestroy(stop));
    }

    return 0;
}
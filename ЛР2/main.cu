#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

double computeHostSum(const std::vector<double>& data) {
    double acc = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        acc += data[i];
    }
    return acc;
}

__global__ void reduceDeviceKernel(const double* d_input, double* d_result, size_t count) {
    double partial = 0.0;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;

    for (size_t i = idx; i < count; i += step) {
        partial += d_input[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(d_result, partial);
    }
}

double computeDeviceSum(const std::vector<double>& h_data, double& out_time_ms) {
    size_t n = h_data.size();
    double* d_vec = nullptr;
    double* d_sum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_vec, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vec, h_data.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t t_start, t_end;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_end));

    CUDA_CHECK(cudaDeviceSynchronize()); 
    CUDA_CHECK(cudaEventRecord(t_start));
    reduceDeviceKernel << <numBlocks, blockSize >> > (d_vec, d_sum, n);
    CUDA_CHECK(cudaEventRecord(t_end));
    CUDA_CHECK(cudaEventSynchronize(t_end));

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t_start, t_end));
    out_time_ms = static_cast<double>(elapsed);

    double h_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_end));

    return h_sum;
}

int main() {
    std::cout.sync_with_stdio(false);
    std::mt19937 rng(42); 
    std::uniform_real_distribution<double> dist(0.0, 10.0);

    {
        double dummy_time = 0.0;
        std::vector<double> warmup(1024, 1.0);
        computeDeviceSum(warmup, dummy_time);
    }

    std::vector<size_t> test_sizes = { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 250000, 500000, 1000000 };

    std::cout << std::left << std::setw(12) << "Elements"
        << std::setw(14) << "CPU[ms]"
        << std::setw(14) << "GPU[ms]"
        << std::setw(10) << "Speedup"
        << "Correct?\n";
    std::cout << std::string(64, '-') << "\n";

    for (size_t n : test_sizes) {
        std::vector<double> dataset(n);
        for (size_t i = 0; i < n; ++i) dataset[i] = dist(rng);

        auto cpu_begin = std::chrono::high_resolution_clock::now();
        double res_cpu = computeHostSum(dataset);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double t_cpu = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

        double t_gpu = 0.0;
        double res_gpu = computeDeviceSum(dataset, t_gpu);

        double accel = (t_gpu > 1e-5) ? t_cpu / t_gpu : 0.0;
        double rel_err = std::abs(res_cpu - res_gpu) / std::abs(res_cpu);
        bool valid = (rel_err < 1e-6);

        std::cout << std::left << std::setw(12) << n
            << std::fixed << std::setprecision(3)
            << std::setw(14) << t_cpu
            << std::setw(14) << t_gpu
            << std::setw(10) << std::setprecision(2) << accel
            << (valid ? "PASS" : "FAIL") << "\n";
    }

    return 0;
}
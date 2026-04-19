#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>

cudaTextureObject_t texObj = 0;

void writefile(float* image, int height, int width, bool gpu = false) {
    unsigned char* output = new unsigned char[width * height];
    for (int i = 0; i < height * width; i++) {
        output[i] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, image[i])));
    }
    stbi_write_bmp(gpu ? "output_gpu1.bmp" : "output_cpu1.bmp", width, height, 1, output);
    delete[] output;
}

__host__ __device__ float g_func(int x, int y, int x_0, int y_0, float sigma_d) {
    return expf(-((x - x_0) * (x - x_0) + (y - y_0) * (y - y_0)) / 2.0f / sigma_d / sigma_d);
}

__host__ __device__ float r_func(float i, float i_0, float sigma_r) {
    return expf(-(i - i_0) * (i - i_0) / 2.0f / sigma_r / sigma_r);
}

void bilateral_filter_cpu(float* output, float* input, int height, int width, float sigma_d, float sigma_r) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float k = 0;
            float h = 0; 
            for (int window_y = i - 1; window_y <= i + 1; ++window_y) {
                for (int window_x = j - 1; window_x <= j + 1; ++window_x) {
                    if (window_y >= 0 && window_y < height && window_x >= 0 && window_x < width) {
                        float w = g_func(window_y, window_x, i, j, sigma_d) *
                            r_func(input[window_y * width + window_x], input[i * width + j], sigma_r);
                        k += w;
                        h += input[window_y * width + window_x] * w;
                    }
                }
            }
            output[i * width + j] = h / k;
        }
    }
}

__global__ void bilateral_filter_gpu(float* output, int height, int width, float sigma_d, float sigma_r,
    cudaTextureObject_t tex) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        float center_val = tex2D<float>(tex, col + 0.5f, row + 0.5f);
        float k = 0;
        float h = 0;

        for (int window_y = -1; window_y <= 1; ++window_y) {
            for (int window_x = -1; window_x <= 1; ++window_x) {
                float curr_i = tex2D<float>(tex, col + window_x + 0.5f, row + window_y + 0.5f);
                float w = g_func(window_y + 1, window_x + 1, 1, 1, sigma_d) *
                    r_func(curr_i, center_val, sigma_r);
                k += w;
                h += curr_i * w;
            }
        }
        output[row * width + col] = h / k;
    }
}

cudaError_t setupTexture(const float* d_data, int width, int height, cudaTextureObject_t* pTexObject) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.devPtr = (void*)d_data;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    return cudaCreateTextureObject(pTexObject, &resDesc, &texDesc, nullptr);
}

void destroyTexture(cudaTextureObject_t texObject) {
    if (texObject != 0) {
        cudaDestroyTextureObject(texObject);
    }
}

int main() {
    int width, height, channels;
    unsigned char* image_data = stbi_load("input1.bmp", &width, &height, &channels, 1); // 1 = grayscale

    if (!image_data) {
        printf("Error: Cannot read input.bmp\n");
        return 1;
    }

    float* image_array = new float[height * width];
    float* outputCPU = new float[height * width];
    float* outputGPU = new float[height * width];

    for (int i = 0; i < height * width; i++) {
        image_array[i] = image_data[i];
    }
    stbi_image_free(image_data);

    printf("Image size: %dx%d\n", width, height);

    float sigma_d, sigma_r;
    std::cout << "Enter sigma_d:\t";
    std::cin >> sigma_d;
    std::cout << "Enter sigma_r:\t";
    std::cin >> sigma_r;

    clock_t start, end;
    start = clock();
    bilateral_filter_cpu(outputCPU, image_array, height, width, sigma_d, sigma_r);
    end = clock();
    float cpu_time = static_cast<float>(end - start) / static_cast<float>(CLOCKS_PER_SEC);
    printf("CPU time: %f sec.\n", cpu_time);

    writefile(outputCPU, height, width, false);

    float* d_image = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_image, height * width * sizeof(float));
    cudaMalloc(&d_output, height * width * sizeof(float));
    cudaMemcpy(d_image, image_array, height * width * sizeof(float), cudaMemcpyHostToDevice);

    cudaTextureObject_t texObj = 0;
    cudaError_t err = setupTexture(d_image, width, height, &texObj);
    if (err != cudaSuccess) {
        printf("Error creating texture: %s\n", cudaGetErrorString(err));
        return 1;
    }

    dim3 block_dim(32, 32);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
        (height + block_dim.y - 1) / block_dim.y);

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin, 0);
    bilateral_filter_gpu << <grid_dim, block_dim >> > (d_output, height, width, sigma_d, sigma_r, texObj);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelErr));
        return 1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_kernel_time;
    cudaEventElapsedTime(&gpu_kernel_time, begin, stop);
    printf("GPU filtering: %f sec.\n", gpu_kernel_time / 1000.0f);

    cudaEventRecord(begin, 0);
    cudaMemcpy(outputGPU, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_memcpy_time;
    cudaEventElapsedTime(&gpu_memcpy_time, begin, stop);

    float total_gpu_time = (gpu_kernel_time + gpu_memcpy_time) / 1000.0f;
    printf("GPU memcpy: %f sec.\n", gpu_memcpy_time / 1000.0f);
    printf("Total GPU time: %f sec.\n", total_gpu_time);
    printf("Speedup: %f\n", cpu_time / total_gpu_time);

    writefile(outputGPU, height, width, true);

    destroyTexture(texObj);
    cudaFree(d_image);
    cudaFree(d_output);
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);

    delete[] image_array;
    delete[] outputCPU;
    delete[] outputGPU;

    return 0;
}
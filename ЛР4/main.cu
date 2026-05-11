#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std;

struct Individual {
    float coeffs[5];
    float fitness;
};

struct DataPoint {
    float x, y;
};

struct FitnessComparator {
    __host__ __device__ bool operator()(const Individual& a, const Individual& b) const {
        return a.fitness < b.fitness;
    }
};



__global__ void setupCurandKernel(curandState* states, unsigned long seed, int numStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void initPopulationKernel(Individual* population, curandState* states, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;

    curandState state = states[idx];
    for (int i = 0; i < 5; ++i) {
        population[idx].coeffs[i] = (curand_uniform(&state) - 0.5f) * 20.0f;
    }
    population[idx].fitness = 1e9f;
    states[idx] = state;
}

__global__ void evaluateFitnessKernel(Individual* population, const DataPoint* data,
    int numPoints, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;

    float maxError = 0.0f;
    for (int i = 0; i < numPoints; ++i) {
        float x = data[i].x;
        float y_true = data[i].y;

        float y_approx = population[idx].coeffs[0] +
            population[idx].coeffs[1] * x +
            population[idx].coeffs[2] * x * x +
            population[idx].coeffs[3] * x * x * x +
            population[idx].coeffs[4] * x * x * x * x;

        float err = fabsf(y_true - y_approx);
        if (err > maxError) maxError = err;
    }
    population[idx].fitness = maxError;
}

__global__ void evolveKernel(Individual* population, curandState* states,
    int popSize, float mutMean, float mutVar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;

    curandState state = states[idx];
    float sigma = sqrtf(mutVar);

    if (idx < popSize / 2) {
        states[idx] = state;
        return;
    }

    int p1 = curand(&state) % (popSize / 2);
    int p2 = (curand(&state) % (popSize / 2) + p1 + 1) % (popSize / 2);
    int crosspoint = 1 + (curand(&state) % 4);

    Individual child;
    for (int i = 0; i < 5; ++i) {
        if (i < crosspoint) {
            child.coeffs[i] = population[p1].coeffs[i];
        }
        else {
            child.coeffs[i] = population[p2].coeffs[i];
        }
    }

    for (int i = 0; i < 5; ++i) {
        if (curand_uniform(&state) < 0.1f) {
            child.coeffs[i] += curand_normal(&state) * sigma + mutMean;
        }
    }

    population[idx] = child;
    population[idx].fitness = 1e9f;
    states[idx] = state;
}



int main() {
    const int NUM_POINTS = 800;
    const int POP_SIZE = 1024;
    const int MAX_GEN = 500;
    const int MAX_CONST_GEN = 60;
    const float MUT_MEAN = 0.0f;
    const float MUT_VAR = 0.25f;
    const unsigned long SEED = 42ULL;

    cout << "Genetic Algorithm: Polynomial Approximation\n";
    cout << "Points: " << NUM_POINTS << " | Population: " << POP_SIZE << "\n";

    vector<DataPoint> h_data(NUM_POINTS);
    float trueCoeffs[5] = { 1.5f, -2.3f, 0.5f, -0.1f, 0.02f };

    for (int i = 0; i < NUM_POINTS; ++i) {
        h_data[i].x = -5.0f + 10.0f * i / (NUM_POINTS - 1);
        float x = h_data[i].x;
        h_data[i].y = trueCoeffs[0] + trueCoeffs[1] * x + trueCoeffs[2] * x * x +
            trueCoeffs[3] * x * x * x + trueCoeffs[4] * x * x * x * x;
        h_data[i].y += ((rand() % 100) / 1000.0f - 0.05f);
    }

    DataPoint* d_data;
    cudaMalloc(&d_data, NUM_POINTS * sizeof(DataPoint));
    cudaMemcpy(d_data, h_data.data(), NUM_POINTS * sizeof(DataPoint), cudaMemcpyHostToDevice);

    Individual* d_population;
    cudaMalloc(&d_population, POP_SIZE * sizeof(Individual));

    curandState* d_states;
    cudaMalloc(&d_states, POP_SIZE * sizeof(curandState));

    int threads = 256;
    int blocks = (POP_SIZE + threads - 1) / threads;
    setupCurandKernel << <blocks, threads >> > (d_states, SEED, POP_SIZE);
    cudaDeviceSynchronize();

    initPopulationKernel << <blocks, threads >> > (d_population, d_states, POP_SIZE);
    cudaDeviceSynchronize();

    cout << "Starting evolution\n";
    auto start = chrono::high_resolution_clock::now();

    float bestFitness = 1e9f;
    int constGenCount = 0;
    int gen = 0;
    Individual h_bestInd;

    vector<Individual> h_population(POP_SIZE);

    for (; gen < MAX_GEN; ++gen) {
        evaluateFitnessKernel << <blocks, threads >> > (d_population, d_data, NUM_POINTS, POP_SIZE);
        cudaDeviceSynchronize();

        cudaMemcpy(h_population.data(), d_population, POP_SIZE * sizeof(Individual), cudaMemcpyDeviceToHost);

        sort(h_population.begin(), h_population.end(), FitnessComparator());

        float currentBest = h_population[0].fitness;
        if (currentBest < bestFitness - 1e-5f) {
            bestFitness = currentBest;
            constGenCount = 0;
            h_bestInd = h_population[0];
        }
        else {
            constGenCount++;
        }

        if (constGenCount >= MAX_CONST_GEN || bestFitness < 0.15f) {
            cout << "Convergence reached at generation " << gen << "\n";
            break;
        }

        cudaMemcpy(d_population, h_population.data(), POP_SIZE * sizeof(Individual), cudaMemcpyHostToDevice);

        evolveKernel << <blocks, threads >> > (d_population, d_states, POP_SIZE, MUT_MEAN, MUT_VAR);
        cudaDeviceSynchronize();
    }

    auto end = chrono::high_resolution_clock::now();
    double gpuTimeSec = chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0;

    cout << "\n========== RESULTS ==========\n";
    cout << "GPU Processing Time: " << gpuTimeSec << " sec\n";
    cout << "Generations Executed: " << gen << "\n";
    cout << "Best Fitness (Max Error): " << bestFitness << "\n";
    cout << "Approximated Coeffs:  ";
    for (int i = 0; i < 5; ++i) cout << h_bestInd.coeffs[i] << " ";
    cout << "\nTrue Coefficients:    ";
    for (int i = 0; i < 5; ++i) cout << trueCoeffs[i] << " ";
    cout << "\n===========================\n";

    cudaFree(d_data);
    cudaFree(d_population);
    cudaFree(d_states);

    FILE* f = fopen("plot_data.csv", "w");
    if (f) {
        fprintf(f, "x,y,approximation\n");
        for (int i = 0; i < NUM_POINTS; i++) {
            float x = -5.0f + 10.0f * i / (NUM_POINTS - 1);
            float y_approx = h_bestInd.coeffs[0] + h_bestInd.coeffs[1] * x +
                h_bestInd.coeffs[2] * x * x + h_bestInd.coeffs[3] * x * x * x +
                h_bestInd.coeffs[4] * x * x * x * x;
            fprintf(f, "%f,%f,%f\n", x, h_data[i].y, y_approx);
        }
        fclose(f);
    }

    return 0;
}
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define TARGET_REPEAT 35 // # of repeats required to stop iterating
#define TIME_STEP 0.01 // a constant that stands for time discretization
#define PRESSURE_SLOPE 0.01 // pumping pressure's linear slope allowing adiabatic evolution
#define DETUNING_FREQUENCY 1 // detuning frequency of the Hamiltonian
#define HEAT_PARAMETER 0.06 // heat parameter for the heated algorithm

__global__ void dot(float* product, float* qubo, float* spin, int dim) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < dim) {
        product[id] = 0;
        for (int i = 0; i < dim; i++) {
            product[id] += qubo[id * dim + i] * (spin[i] > 0 ? 1 : (spin[i] < 0 ? -1 : 0));
        }
    }
}

__global__ void initRand(float* arr, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < size) {
        curandState state;
        curand_init(clock64(), id, 0, &state);
        arr[id] = 2 * curand_uniform(&state) - 1;
    }
}

/**
 * @brief calculate the standard deviation of given array
 *
 * @param arr input array
 * @param size size of the array
 * @return the standard deviation of the array
 */
float stddiv(float* arr, int size) {
    float mean = 0;
    for (int i = 0; i < size; i++) {
        mean += arr[i];
    }
    mean /= size;
    float sumDistanceSquare = 0;
    for (int i = 0; i < size; i++) {
        sumDistanceSquare += pow(arr[i] - mean, 2);
    }
    return sqrt(sumDistanceSquare / size);
}

/**
 * @brief update the spin and its momemtum
 *
 * @param spin the spin array
 * @param momentum momentum of the spin
 * @param qubo the relationship matrix parsed to 1D
 * @param dim size of the array
 * @param step the # of step
 * @param xi0 a constant calculated with qubo matrix
 */
__global__ void update(float* spin, float* momentum, float* dot_product, int dim, int step, float xi0) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < dim) {
        float pressure = PRESSURE_SLOPE * TIME_STEP * step;
        momentum[id] += TIME_STEP * ((pressure - DETUNING_FREQUENCY) * spin[id] + xi0 * dot_product[id]);
        spin[id] += TIME_STEP * DETUNING_FREQUENCY * momentum[id];
    }
    // float pressure = PRESSURE_SLOPE * TIME_STEP * step;

    // for (int i = 0; i < dim; i++) {
    //     float dot_product = 0;
    //     for (int j = 0; j < dim; j++) {
    //         dot_product += qubo[i * dim + j] * (spin[j] > 0 ? 1 : (spin[j] < 0 ? -1 : 0));
    //     }
    //     momentum[i] += TIME_STEP * ((pressure - DETUNING_FREQUENCY) * spin[i] + xi0 * dot_product);
    //     spin[i] += TIME_STEP * DETUNING_FREQUENCY * momentum[i];
    //     // printf("%s", spin[i] > 0 ? "+" : "-");
    // }
    // // printf("\n");


}

/**
 * @brief bound spins within the range [-1,1]
 *
 * @param spin the spin array
 * @param momentum the momentum of the array
 * @param dim size of the spin array
 */
__global__ void confine(float* spin, float* momentum, int dim) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < dim) {
        if (spin[id] < -1) {
            spin[id] = -1;
            momentum[id] = 0;
        } else if (spin[id] > 1) {
            spin[id] = 1;
            momentum[id] = 0;
        }
    }
}

/**
 * @brief calculate the heated momentum of the spins
 *
 * @param momentum the momentum of the spin
 * @param pastMomentum the previous momentum of the spin
 * @param dim the size of the spin array
 */
__global__ void heatUp(float* momentum, float* pastMomentum, int dim) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < dim) {
        momentum[id] += pastMomentum[id] * TIME_STEP * HEAT_PARAMETER;
    }
}

/**
 * @brief check if the given two spin array have same spin
 *
 * @param spin1 first spin array
 * @param spin2 second spin array
 * @param dim size of spin array
 * @return 0 if true, else false
 */
int sameSpin(float* spin1, float* spin2, int dim) {
    int sameCount = 0;
    for (int i = 0; i < dim; i++) {
        sameCount += spin1[i] * spin2[i] > 0 ? 1 : 0;
    }
    // printf("--not same count: %d--", dim - sameCount);
    return dim - sameCount;
}

extern "C" {
    void iterate(float* spin, float* qubo, int dim, int window, int maxStep);
}

/**
 * @brief the iteration step of the simulated bifurcation algorithm
 *
 * @param spin the spin array
 * @param qubo the relationship matrix pased into 1D
 * @param dim dimention of the spin array
 * @param window number of time steps between two spin sampling. if 0 then no window used
 * @param maxStep maximum iteration of the algorithm
 */
void iterate(float* spin, float* qubo, int dim, int window, int maxStep) {

    if (window < 0) {
        printf("window cannot be negative.\n");
        exit(-1);
    }

    float* momentum;
    cudaMalloc(&momentum, dim * sizeof(float));
    initRand << <50, 50 >> > (momentum, dim);
    float* pastMomentum;
    cudaMalloc(&pastMomentum, dim * sizeof(float));
    cudaDeviceSynchronize();

    float xi0;
    xi0 = (0.7 * DETUNING_FREQUENCY) / stddiv(qubo, dim * dim) * sqrt(dim);

    float** sample;
    if (window != 0) {
        sample = (float**)malloc((maxStep / window + 1) * sizeof(float*));
        for (int i = 0; i < maxStep / window + 1; i++) {
            sample[i] = (float*)malloc(dim * sizeof(float));
        }
    }

    float* dot_product;
    cudaMalloc(&dot_product, dim * sizeof(float));

    float* spin_dev;
    cudaMalloc(&spin_dev, dim * sizeof(float));
    cudaMemcpy(spin_dev, spin, dim * sizeof(float), cudaMemcpyHostToDevice);

    float* qubo_dev;
    cudaMalloc(&qubo_dev, dim * dim * sizeof(float));
    cudaMemcpy(qubo_dev, qubo, dim * dim * sizeof(float), cudaMemcpyHostToDevice);

    if (window == 0) {
        for (int i = 0; i < maxStep; i++) {
            cudaMemcpy(pastMomentum, momentum, dim * sizeof(float), cudaMemcpyDeviceToDevice);
            dot << <50, 50 >> > (dot_product, qubo_dev, spin_dev, dim);
            cudaDeviceSynchronize();
            update << <50, 50 >> > (spin_dev, momentum, dot_product, dim, i, xi0);
            //cudaDeviceSynchronize();
            confine << <50, 50 >> > (spin_dev, momentum, dim);
            //cudaDeviceSynchronize();
            heatUp << <50, 50 >> > (momentum, pastMomentum, dim);
            cudaDeviceSynchronize();
        }
    } else {
        int repeatNum = 0;
        for (int i = 0; i < maxStep; i++) {
            cudaMemcpy(pastMomentum, momentum, dim * sizeof(float), cudaMemcpyDeviceToDevice);
            dot << <50, 50 >> > (dot_product, qubo_dev, spin_dev, dim);
            cudaDeviceSynchronize();
            update << <50, 50 >> > (spin_dev, momentum, dot_product, dim, i, xi0);
            // cudaDeviceSynchronize();
            confine << <50, 50 >> > (spin_dev, momentum, dim);
            // cudaDeviceSynchronize();
            heatUp << <50, 50 >> > (momentum, pastMomentum, dim);
            if (i % window == 0) {
                cudaMemcpy(sample[i / window], spin_dev, dim * sizeof(float), cudaMemcpyDeviceToHost);
                if (i != 0) {
                    sameSpin(sample[i / window], sample[i / window - 1], dim) == 0 ? (repeatNum++) : (repeatNum = 0);
                    if (repeatNum == TARGET_REPEAT) {
                        // printf("meet criteria at step = %d\n", i);
                        break;
                    }
                }
            }
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(spin, spin_dev, dim * sizeof(float), cudaMemcpyDeviceToHost);

    if (window != 0) {
        for (int i = 0; i < maxStep / window + 1; i++) {
            free(sample[i]);
        }
        free(sample);
    }
    cudaFree(spin_dev);
    cudaFree(qubo_dev);
    cudaFree(dot_product);
    cudaFree(momentum);
    cudaFree(pastMomentum);
}

int main() {

    float spin[] = { 0.5f,-0.5f };
    float qubo[] = { 0.0f, 1.0f, 1.0f, 0.0f };
    int dim = 2;
    int window = 0;
    int maxStep = 200;
    iterate(spin, qubo, dim, window, maxStep);

    for (int i = 0; i < dim; i++) {
        printf("%s", spin[i] > 0 ? "+" : "-");
    }
    printf("\n");
}
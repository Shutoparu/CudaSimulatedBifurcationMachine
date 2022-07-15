#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define TARGET_REPEAT 50 // # of repeats required to stop iterating
#define TIME_STEP 0.01 // a constant that stands for time discretization
#define PRESSURE_SLOPE 0.01 // pumping pressure's linear slope allowing adiabatic evolution
#define DETUNING_FREQUENCY 1 // detuning frequency of the Hamiltonian
#define HEAT_PARAMETER 0.06 // heat parameter for the heated algorithm

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
void update(float* spin, float* momentum, float* qubo, int dim, int step, float xi0) {
    float pressure = PRESSURE_SLOPE * TIME_STEP * step;

    for (int i = 0; i < dim; i++) {
        float dot_product = 0;
        for (int j = 0; j < dim; j++) {
            dot_product += qubo[i * dim + j] * (spin[j] > 0 ? 1 : (spin[j] < 0 ? -1 : 0));
        }
        momentum[i] += TIME_STEP * ((pressure - DETUNING_FREQUENCY) * spin[i] + xi0 * dot_product);
        spin[i] += TIME_STEP * DETUNING_FREQUENCY * momentum[i];
        // printf("%s", spin[i] > 0 ? "+" : "-");
    }
    // printf("\n");


}

/**
 * @brief bound spins within the range [-1,1]
 *
 * @param spin the spin array
 * @param momentum the momentum of the array
 * @param dim size of the spin array
 */
void confine(float* spin, float* momentum, int dim) {
    for (int i = 0; i < dim; i++) {
        if (spin[i] < -1) {
            spin[i] = -1;
            momentum[i] = 0;
        } else if (spin[i] > 1) {
            spin[i] = 1;
            momentum[i] = 0;
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
void heatUp(float* momentum, float* pastMomentum, int dim) {
    for (int i = 0; i < dim; i++) {
        momentum[i] += pastMomentum[i] * TIME_STEP * HEAT_PARAMETER;
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

/**
 * @brief the iteration step of the simulated bifurcation algorithm
 *
 * @param spin the spin array
 * @param qubo the relationship matrix pased into 1D
 * @param dim dimention of the spin array
 * @param window number of time steps between two spin sampling. if 0 then no window used
 * @param maxStep maximum iteration of the algorithm
 */
extern void iterate(float* spin, float* qubo, int dim, int window, int maxStep) {

    if (window < 0) {
        printf("window cannot be negative.\n");
        exit(-1);
    }

    float* momentum;
    momentum = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        momentum[i] = 2 * (rand() / (float)RAND_MAX) - 1;
    }
    float* pastMomentum;
    pastMomentum = (float*)malloc(dim * sizeof(float));

    float xi0;
    xi0 = (0.7 * DETUNING_FREQUENCY) / stddiv(qubo, dim * dim) * sqrt(dim);

    float** sample;
    if (window != 0) {
        sample = (float**)malloc((maxStep / window + 1) * sizeof(float*));
        for (int i = 0; i < maxStep / window + 1; i++) {
            sample[i] = (float*)malloc(dim * sizeof(float));
        }
    }

    if (window == 0) {
        for (int i = 0; i < maxStep; i++) {
            memcpy(pastMomentum, momentum, dim * sizeof(float));
            update(spin, momentum, qubo, dim, i, xi0);
            confine(spin, momentum, dim);
            heatUp(momentum, pastMomentum, dim);
        }
    } else {
        int repeatNum = 0;
        for (int i = 0; i < maxStep; i++) {
            memcpy(pastMomentum, momentum, dim * sizeof(float));
            update(spin, momentum, qubo, dim, i, xi0);
            confine(spin, momentum, dim);
            heatUp(momentum, pastMomentum, dim);
            if (i % window == 0) {
                memcpy(sample[i / window], spin, dim * sizeof(float));
                if (i != 0) {
                    sameSpin(sample[i / window], sample[i / window - 1], dim) == 0 ? (repeatNum++) : (repeatNum = 0);
                    // printf("\trepeatNum: %d\n", repeatNum);
                    if (repeatNum == TARGET_REPEAT) {
                        printf("meet criteria at step = %d\n", i);
                        break;
                    }
                }
            }
        }
    }
    if (window != 0) {
        for (int i = 0; i < maxStep / window + 1; i++) {
            free(sample[i]);
        }
        free(sample);
    }
    free(momentum);
    free(pastMomentum);
}

// int main(){

//     iterate(float* spin, float* qubo, int dim, int window, int maxStep)
// }
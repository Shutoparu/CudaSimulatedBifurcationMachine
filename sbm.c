#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define TARGET_REPEAT 100 // i
#define TIME_STEP 0.01
#define PRESSURE_SLOPE 0.01
#define MAX_PRESSURE 1
#define DETUNING_FREQUENCY 1
#define HEAT_PARAMETER 0.06

/**
 * @brief
 *
 * @param arr
 * @param size
 * @return float
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
 * @brief
 *
 * @param spin
 * @param momentum
 * @param qubo
 * @param dim
 * @param step
 * @param xi0
 */
void update(float* spin, float* momentum, float* qubo, int dim, int step, float xi0) {
    float pressure = PRESSURE_SLOPE * TIME_STEP * step;
    // float pressure = MAX_PRESSURE;
    // pressure = pressure > MAX_PRESSURE ? MAX_PRESSURE : pressure;
    for (int i = 0; i < dim; i++) {
        float dot_product = 0;
        for (int j = 0; j < dim; j++) {
            dot_product += qubo[i * dim + j] * (spin[j] > 0 ? 1 : (spin[j] < 0 ? -1 : 0));
        }
        momentum[i] += TIME_STEP * ((pressure - DETUNING_FREQUENCY) * spin[i] + xi0 * dot_product);
        spin[i] += TIME_STEP * DETUNING_FREQUENCY * momentum[i];
    }
}

/**
 * @brief
 *
 * @param spin
 * @param dim
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
 * @brief
 *
 * @param momentum
 * @param pastMomentum
 * @param dim
 */
void heatUp(float* momentum, float* pastMomentum, int dim) {
    for (int i = 0; i < dim; i++) {
        momentum[i] += pastMomentum[i] * TIME_STEP * HEAT_PARAMETER;
    }
}

/**
 * @brief
 *
 * @param spin1
 * @param spin2
 * @param dim
 * @return 0 if true, else false
 */
int sameSpin(float* spin1, float* spin2, int dim) {
    int sameCount = 0;
    for (int i = 0; i < dim; i++) {
        sameCount += spin1[i] * spin2[i] > 0 ? 1 : 0;
    }
    printf("--not same count: %d--", dim - sameCount);
    return dim - sameCount;
}

/**
 * @brief
 *
 * @param spin
 * @param dim
 * @param window
 * @param maxStep
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
                    printf("\trepeatNum: %d\n", repeatNum);
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
}

// int main(){

//     iterate(float* spin, float* qubo, int dim, int window, int maxStep)
// }
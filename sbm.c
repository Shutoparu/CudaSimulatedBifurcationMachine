#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define TARGET_REPEAT 50
#define TIME_STEP 0.01
#define PRESSURE_SLOPE 0.01
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
    for (int i = 0; i < dim; i++) {
        momentum[i] += TIME_STEP * (PRESSURE_SLOPE * TIME_STEP * step - DETUNING_FREQUENCY) * spin[i];
        spin[i] += TIME_STEP * (PRESSURE_SLOPE * TIME_STEP * step + DETUNING_FREQUENCY - xi0 * qubo[i * dim + i]) * momentum[i];

        float dot_product = 0;
        for (int j = 0; j < dim; j++) {
            dot_product += qubo[i * dim + j] * spin[j];
        }
        momentum[i] += TIME_STEP * (xi0 * dot_product + HEAT_PARAMETER * momentum[i]);
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
    return sameCount - dim;
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

    float xi0;
    xi0 = (0.7 * DETUNING_FREQUENCY) / stddiv(qubo, dim * dim) * sqrt(dim);

    float** sample;
    if (window != 0) {
        sample = (float**)malloc((maxStep / window + 1) * sizeof(float*));
        for (int i = 0; i < maxStep / window + 1; i++) {
            sample[i] = (float*)malloc(dim * sizeof(float));
        }
    }

    int repeatNum = 0;

    if (window == 0) {
        for (int i = 0; i < maxStep; i++) {
            update(spin, momentum, qubo, dim, i, xi0);
            confine(spin, momentum, dim);
        }
    } else {
        for (int i = 0; i < maxStep; i++) {
            update(spin, momentum, qubo, dim, i, xi0);
            confine(spin, momentum, dim);
            if (i % window == 0) {
                memcpy(sample[i / window], spin, dim * sizeof(float));
                if (i != 0) {
                    int checkRepeat = sameSpin(sample[i / window], sample[i / window - 1], dim);
                    repeatNum += sameSpin == 0 ? 1 : -repeatNum;
                    if (repeatNum == TARGET_REPEAT) {
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
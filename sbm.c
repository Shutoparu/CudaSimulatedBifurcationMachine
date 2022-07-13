#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define TARGET_REPEAT 50


void update() {

}

/**
 * @brief 
 * 
 * @param spin 
 * @param dim 
 */
void confine(float* spin, int dim) {
    for (int i = 0; i < dim; i++) {
        if (spin[i] < -1) {
            spin[i] = -1;
        } else if (spin[i] > 1) {
            spin[i] = 1;
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
void iterate(float* spin, int dim, int window, int maxStep) {

    if (window < 0) {
        printf("window cannot be negative.\n");
        exit(-1);
    }

    switch (window) {
    case 0:
        for (int i = 0; i < maxStep; i++) {
            update();
            confine(spin, dim);
        }
        break;
    default:
        float** sample;
        sample = (float**)malloc((maxStep / window + 1) * sizeof(float*));
        for (int i = 0; i < maxStep / window + 1; i++) {
            sample[i] = (float*)malloc(dim * sizeof(float));
        }

        int repeatNum = 0;
        for (int i = 0; i < maxStep; i++) {
            update();
            confine(spin, dim);
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

        for (int i = 0; i < maxStep / window + 1; i++) {
            free(sample[i]);
        }
        free(sample);
        break;
    }
}
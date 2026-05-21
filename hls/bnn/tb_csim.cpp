#include "cnn_forward.h"

#include <cstdio>

#include "images.h"

int main() {
    int correct = 0;
    const int n = IMAGE_NUM < 280 ? IMAGE_NUM : 280;

    printf("MNIST BNN C-sim — %d images from images.h\n", n);
    printf("label = i %% 10  (interleaved digit order from hls_export)\n\n");

    for (int i = 0; i < n; ++i) {
        float input[CNN_IN_C][CNN_IN_H][CNN_IN_W];
        float logits[CNN_NUM_CLASS];

        for (int h = 0; h < CNN_IN_H; ++h) {
            for (int w = 0; w < CNN_IN_W; ++w) {
                input[0][h][w] = IMAGES[i][0][h][w];
            }
        }

        const int pred = mnist_bnn_predict(input, logits);
        const int label = i % 10;

        if (pred == label) ++correct;

        if (i < 10 || pred != label) {
            printf("img %3d  label=%d  pred=%d  %s  logits[0..3]=%.3f %.3f %.3f %.3f\n",
                   i, label, pred, pred == label ? "OK" : "NG",
                   logits[0], logits[1], logits[2], logits[3]);
        }
    }

    const float acc = 100.0f * correct / n;
    printf("\nAccuracy: %d / %d = %.2f%%\n", correct, n, acc);
    return acc >= 90.0f ? 0 : 1;
}

#include <iostream>
#include <cstdlib>
#include "omp.h"
#include <cmath>
#include <chrono>

using namespace std;

#define RUN_TYPE static

FILE *inputPicture;
unsigned int count;
float coefficient;
unsigned int val;

unsigned int getInt() {
    unsigned int ret = 0;
    char ch;
    fscanf(inputPicture, "%c", &ch);
    while (!isspace(ch)) {
        ret *= 10;
        ret += (ch - '0');
        fscanf(inputPicture, "%c", &ch);
    }
    return ret;
}

pair<unsigned char, unsigned char> getLowHigh(int a[]) {
    int high = 255;
    unsigned int sum = 0;
    for (; high >= 0; high--) {
        if (sum + a[high] > val) {
            break;
        }
        sum += a[high];
    }
    int low = 0;
    sum = 0;
    for (; low <= 255; low++) {
        if (sum + a[low] > val) {
            break;
        }
        sum += a[low];
    }
    return {low, high};
}

int main(int argc, char *argv[]) {
    int numThreads = stoi(argv[1]);
    inputPicture = fopen(argv[2], "rb");
    unsigned char type;
    char ch;
    fscanf(inputPicture, "%c%c", &type, &type);
    fscanf(inputPicture, "%c", &ch);
    unsigned int width = getInt();
    unsigned int prevWidth = width;
    unsigned int height = getInt();
    unsigned int maxValue = getInt();
    coefficient = stof(argv[4]);
    count = width * height;
    val = round(count * coefficient);
    if (type == '6') {
        width *= 3;
        count *= 3;
    }
    auto *picture = (unsigned char *) malloc(width * height * sizeof(unsigned char *));
    fread(picture, 1, count, inputPicture);
    fclose(inputPicture);
    auto begin = std::chrono::high_resolution_clock::now();
    if (type == '5') {
        int grey[256] = {0};
        #pragma omp parallel shared(grey, count) num_threads(numThreads)
        {
            int localgrey[256] = {0};
            #pragma omp for schedule(RUN_TYPE)
            for (int i = 0; i < count; i++) {
                localgrey[picture[i]]++;
            }
            #pragma omp critical
            for (int i = 0; i < 256; i++) {
                grey[i] += localgrey[i];
            }
        }
        pair<unsigned char, unsigned char> pair1 = getLowHigh(grey);
        unsigned char low = pair1.first;
        unsigned char high = pair1.second;
        unsigned char dop = (high - low) / 2;
        #pragma omp parallel for num_threads(numThreads) schedule(RUN_TYPE)
        for (int i = 0; i < count; i++) {
            if (picture[i] >= high) {
                picture[i] = 255;
                continue;
            }
            if (picture[i] <= low) {
                picture[i] = 0;
                continue;
            }
            picture[i] = (unsigned char) ((picture[i] - low) * 255 + dop) / (high - low);
        }
    } else {
        int rgb[3][256];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 256; j++) {
                rgb[i][j] = 0;
            }
        }
        #pragma omp parallel shared(rgb, count) num_threads(numThreads)
        {
            int localrgb[3][256];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 256; j++) {
                    localrgb[i][j] = 0;
                }
            }
            #pragma omp for schedule(RUN_TYPE)
            for (int i = 0; i < count; i++) {
                localrgb[i % 3][picture[i]]++;
            }
            #pragma omp critical
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 256; j++) {
                    rgb[i][j] += localrgb[i][j];
                }
            }
        }
        pair<unsigned char, unsigned char> lowHighRed;
        pair<unsigned char, unsigned char> lowHighGreen;
        pair<unsigned char, unsigned char> lowHighBlue;
        #pragma omp parallel sections num_threads(numThreads)
        {
            #pragma omp section
            lowHighRed = getLowHigh(rgb[0]);
            #pragma omp section
            lowHighGreen = getLowHigh(rgb[1]);
            #pragma omp section
            lowHighBlue = getLowHigh(rgb[2]);
        }
        unsigned char lowRGB = min(lowHighRed.first, min(lowHighGreen.first, lowHighBlue.first));
        unsigned char highRGB = max(lowHighRed.second, max(lowHighGreen.second, lowHighBlue.second));
        int kRGB = highRGB - lowRGB;
        unsigned char dop = kRGB / 2;
        #pragma omp parallel for num_threads(numThreads) schedule(RUN_TYPE)
        for (int i = 0; i < count; i++) {
            if (picture[i] >= highRGB) {
                picture[i] = 255;
                continue;
            }
            if (picture[i] <= lowRGB) {
                picture[i] = 0;
                continue;
            }
            picture[i] = (unsigned char) ((((picture[i] - lowRGB) * 255 + dop) / kRGB));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    FILE *outputPicture = fopen(argv[3], "wb");
    fprintf(outputPicture, "P%c\n%d %d\n%d\n", type, prevWidth, height, maxValue);
    fwrite(picture, 1, count, outputPicture);
    fclose(outputPicture);
    printf("Time (%i thread(s)): %g ms\n", numThreads, duration);
}
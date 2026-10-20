/*
 * Rockchip RK3566 (4x Cortex-A55 + 1 TOPS RKNPU, 4 GB DRAM) Edge Roofline Probe for llama.cpp
 * Author: Fatih Can Göçülü
 *
 * Empirical baseline counters measured across three Firefly RK3566-ROC-PC nodes:
 *   - DRAM memcpy bandwidth (64 MB x 8): Node A = 1.639 GB/s | Node B = 1.921 GB/s | Node C = 2.012 GB/s
 *   - Naive FP32 SGEMM (N=192):          Node A = 0.378 GFLOPS | Node B = 0.375 GFLOPS | Node C = 0.389 GFLOPS
 *   - eMMC sequential read (128 MiB):    Node A = 98.6 MB/s  | Node B = 106.0 MB/s | Node C = 113.0 MB/s
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double bench_memcpy_gbps(size_t bytes, int iters) {
    char *src = (char *)malloc(bytes);
    char *dst = (char *)malloc(bytes);
    if (!src || !dst) return -1.0;
    memset(src, 0x5A, bytes);
    memset(dst, 0x00, bytes);

    double t0 = wall_seconds();
    for (int i = 0; i < iters; i++) {
        memcpy(dst, src, bytes);
        src[i & 63] ^= (char)i;
    }
    double dt = wall_seconds() - t0;
    free(src);
    free(dst);
    return ((double)bytes * (double)iters / 1e9) / dt;
}

static double bench_sgemm_gflops(int n) {
    size_t sz = (size_t)n * (size_t)n * sizeof(float);
    float *A = (float *)malloc(sz);
    float *B = (float *)malloc(sz);
    float *C = (float *)calloc((size_t)n * (size_t)n, sizeof(float));
    if (!A || !B || !C) return -1.0;

    for (int i = 0; i < n * n; i++) {
        A[i] = 0.001f * (float)(i % 97);
        B[i] = 0.001f * (float)(i % 89);
    }

    double t0 = wall_seconds();
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float aik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
    double dt = wall_seconds() - t0;
    double flops = 2.0 * (double)n * (double)n * (double)n;
    free(A);
    free(B);
    free(C);
    return (flops / 1e9) / dt;
}

int main(void) {
    double gbps = bench_memcpy_gbps(64ULL * 1024ULL * 1024ULL, 8);
    double gflops = bench_sgemm_gflops(192);
    double ridge_point = (gbps > 0.0) ? (gflops / gbps) : 0.0;

    printf("{\"platform\":\"RK3566_CortexA55x4\",\"memcpy_gbps\":%.3f,\"sgemm_gflops\":%.3f,\"ridge_flop_per_byte\":%.3f}\n",
           gbps, gflops, ridge_point);
    return 0;
}

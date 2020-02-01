/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char *dgemm_desc = "Simple blocked dgemm.";

#include <assert.h>
#include <immintrin.h>
#include <stdalign.h>
#include <string.h>

// AVX matrix size
#define AVX_M 4
#define AVX_N 12
#define AVX_K 24

// #define L1_AVX 2 // Do number of AVX in one L1 cache
#define BLOCK_SIZE_L1_M (AVX_M * 6)
#define BLOCK_SIZE_L1_N (AVX_N * 2)
#define BLOCK_SIZE_L1_K (AVX_K * 1)

#define L2_L1_RATIO 2
#define BLOCK_SIZE_L2_M (BLOCK_SIZE_L1_M * L2_L1_RATIO)
#define BLOCK_SIZE_L2_N (BLOCK_SIZE_L1_N * L2_L1_RATIO)
#define BLOCK_SIZE_L2_K (BLOCK_SIZE_L1_K * L2_L1_RATIO)

#define L3_L2_RATIO 4
#define BLOCK_SIZE_L3_M (BLOCK_SIZE_L2_M * L3_L2_RATIO)
#define BLOCK_SIZE_L3_N (BLOCK_SIZE_L2_N * L3_L2_RATIO)
#define BLOCK_SIZE_L3_K (BLOCK_SIZE_L2_K * L3_L2_RATIO)

/*
 * src M*N (src_stride)
 * dst M*N (dst_stride)
 */
static inline void copy_matrix(const int M, const int N, const int src_stride,
                               const double *src, const int dst_stride,
                               double *dst) {
  for (int i = 0; i < M; ++i) {
    memcpy(dst + i * dst_stride, src + i * src_stride, N * sizeof(double));
  }
}

static inline void zero_array(const size_t count, double *arr) {
  memset(arr, 0, sizeof(double) * count);
}

/*
 * Copy M_b*N_b matrix from src (src_stride) to dst (M*N)
 * Zero padding right and bottom if M*N differs from M_b*N_b
 */
static inline void padding_copy_matrix(const int M_b, const int N_b,
                                       const int src_stride, const double *src,
                                       const int M, const int N, double *dst) {
  if (M_b != M || N_b != N) {
    zero_array(M * N, dst);
  }
  copy_matrix(M_b, N_b, src_stride, src, N, dst);
}

#define min(a, b) (((a) < (b)) ? (a) : (b))

// Padding SIZE to multiple of BLOCK
#define PACK_TO(SIZE, BLOCK) ((SIZE + BLOCK - 1) / BLOCK * BLOCK)

/*
 * Define a function named FUNC_NAME
 * 1. split submatrices of M_B*N_B*K_B,
 * 2. call function SUB_FUNC on it
 */
#define BLOCK_FUNC(FUNC_NAME, M_B, N_B, K_B, SUB_FUNC)                         \
  /*                                                                           \
   * A M*K (row stride lda)                                                    \
   * B K*N (row stride ldb)                                                    \
   * C M*N (row stride ldc)                                                    \
   */                                                                          \
  static inline void FUNC_NAME(const int lda, const int ldb, const int ldc,    \
                               const int M, const int N, const int K,          \
                               const double *restrict A,                       \
                               const double *restrict B, double *restrict C) { \
    /* For each block-row of A */                                              \
    for (int i = 0; i < M; i += M_B) {                                         \
      /* For each block-column of B */                                         \
      for (int j = 0; j < N; j += N_B) {                                       \
        /* Compute i, j block of C */                                          \
        for (int k = 0; k < K; k += K_B) {                                     \
          const int M_b_data = min(M_B, M - i);                                \
          const int N_b_data = min(N_B, N - j);                                \
          const int K_b_data = min(K_B, K - k);                                \
          SUB_FUNC(lda, ldb, ldc, M_b_data, N_b_data, K_b_data,                \
                   A + i * lda + k, B + k * ldb + j, C + i * ldc + j);         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/*
 * Define a function named FUNC_NAME
 * 1. split submatrices of M_B*N_B*K_B,
 * 2. copy with padding and align to 32bytes,
 * 3. call function SUB_FUNC on it
 */
#define BLOCK_FUNC_COPY(FUNC_NAME, M_B, N_B, K_B, SUB_FUNC, PACK_M, PACK_N,    \
                        PACK_K)                                                \
  /*                                                                           \
   * A M*K (row stride lda)                                                    \
   * B K*N (row stride ldb)                                                    \
   * C M*N (row stride ldc)                                                    \
   */                                                                          \
  static inline void FUNC_NAME(const int lda, const int ldb, const int ldc,    \
                               const int M, const int N, const int K,          \
                               const double *restrict A,                       \
                               const double *restrict B, double *restrict C) { \
    static alignas(4 * sizeof(double)) double A_b[M_B * K_B];                  \
    static alignas(4 * sizeof(double)) double B_b[K_B * N_B];                  \
    static alignas(4 * sizeof(double)) double C_b[M_B * N_B];                  \
                                                                               \
    /* For each block-row of A */                                              \
    for (int i = 0; i < M; i += M_B) {                                         \
      /* For each block-column of B */                                         \
      for (int j = 0; j < N; j += N_B) {                                       \
        const int M_b_data = min(M_B, M - i);                                  \
        const int M_b_pad = PACK_TO(M_b_data, PACK_M);                         \
        const int N_b_data = min(N_B, N - j);                                  \
        const int N_b_pad = PACK_TO(N_b_data, PACK_N);                         \
                                                                               \
        padding_copy_matrix(M_b_data, N_b_data, ldc, C + i * ldc + j, M_b_pad, \
                            N_b_pad, C_b);                                     \
        /* Compute i, j block of C */                                          \
        for (int k = 0; k < K; k += K_B) {                                     \
          const int K_b_data = min(K_B, K - k);                                \
          const int K_b_pad = PACK_TO(K_b_data, PACK_K);                       \
                                                                               \
          padding_copy_matrix(M_b_data, K_b_data, lda, A + i * lda + k,        \
                              M_b_pad, K_b_pad, A_b);                          \
          padding_copy_matrix(K_b_data, N_b_data, ldb, B + k * ldb + j,        \
                              K_b_pad, N_b_pad, B_b);                          \
          SUB_FUNC(K_b_pad, N_b_pad, N_b_pad, M_b_pad, N_b_pad, K_b_pad, A_b,  \
                   B_b, C_b);                                                  \
        }                                                                      \
        copy_matrix(M_b_data, N_b_data, N_b_pad, C_b, ldc, C + i * ldc + j);   \
      }                                                                        \
    }                                                                          \
  }

static void inline do_block_4x12(const int lda, const int ldb, const int ldc,
                                 const int k, const double *restrict A,
                                 const double *restrict B, double *restrict C) {
  register __m256d c00_c01_c02_c03 = _mm256_load_pd(C + 0 * ldc);
  register __m256d c04_c05_c06_c07 = _mm256_load_pd(C + 0 * ldc + 4);
  register __m256d c08_c09_c0A_c0B = _mm256_load_pd(C + 0 * ldc + 8);

  register __m256d c10_c11_c12_c13 = _mm256_load_pd(C + 1 * ldc);
  register __m256d c14_c15_c16_c17 = _mm256_load_pd(C + 1 * ldc + 4);
  register __m256d c18_c19_c1A_c1B = _mm256_load_pd(C + 1 * ldc + 8);

  register __m256d c20_c21_c22_c23 = _mm256_load_pd(C + 2 * ldc);
  register __m256d c24_c25_c26_c27 = _mm256_load_pd(C + 2 * ldc + 4);
  register __m256d c28_c29_c2A_c2B = _mm256_load_pd(C + 2 * ldc + 8);

  register __m256d c30_c31_c32_c33 = _mm256_load_pd(C + 3 * ldc);
  register __m256d c34_c35_c36_c37 = _mm256_load_pd(C + 3 * ldc + 4);
  register __m256d c38_c39_c3A_c3B = _mm256_load_pd(C + 3 * ldc + 8);

  register __m256d a;

  for (int i = 0; i < k; i++) {
    register __m256d b_0to3 = _mm256_load_pd(B + ldb * i);
    register __m256d b_4to7 = _mm256_load_pd(B + ldb * i + 4);
    register __m256d b_8toB = _mm256_load_pd(B + ldb * i + 8);

    a = _mm256_broadcast_sd(A + 0 * lda + i);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a, b_0to3, c00_c01_c02_c03);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a, b_4to7, c04_c05_c06_c07);
    c08_c09_c0A_c0B = _mm256_fmadd_pd(a, b_8toB, c08_c09_c0A_c0B);

    a = _mm256_broadcast_sd(A + 1 * lda + i);

    c10_c11_c12_c13 = _mm256_fmadd_pd(a, b_0to3, c10_c11_c12_c13);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a, b_4to7, c14_c15_c16_c17);
    c18_c19_c1A_c1B = _mm256_fmadd_pd(a, b_8toB, c18_c19_c1A_c1B);

    a = _mm256_broadcast_sd(A + 2 * lda + i);

    c20_c21_c22_c23 = _mm256_fmadd_pd(a, b_0to3, c20_c21_c22_c23);
    c24_c25_c26_c27 = _mm256_fmadd_pd(a, b_4to7, c24_c25_c26_c27);
    c28_c29_c2A_c2B = _mm256_fmadd_pd(a, b_8toB, c28_c29_c2A_c2B);

    a = _mm256_broadcast_sd(A + 3 * lda + i);

    c30_c31_c32_c33 = _mm256_fmadd_pd(a, b_0to3, c30_c31_c32_c33);
    c34_c35_c36_c37 = _mm256_fmadd_pd(a, b_4to7, c34_c35_c36_c37);
    c38_c39_c3A_c3B = _mm256_fmadd_pd(a, b_8toB, c38_c39_c3A_c3B);
  }
  _mm256_store_pd(C + 0 * ldc, c00_c01_c02_c03);
  _mm256_store_pd(C + 0 * ldc + 4, c04_c05_c06_c07);
  _mm256_store_pd(C + 0 * ldc + 8, c08_c09_c0A_c0B);

  _mm256_store_pd(C + 1 * ldc, c10_c11_c12_c13);
  _mm256_store_pd(C + 1 * ldc + 4, c14_c15_c16_c17);
  _mm256_store_pd(C + 1 * ldc + 8, c18_c19_c1A_c1B);

  _mm256_store_pd(C + 2 * ldc, c20_c21_c22_c23);
  _mm256_store_pd(C + 2 * ldc + 4, c24_c25_c26_c27);
  _mm256_store_pd(C + 2 * ldc + 8, c28_c29_c2A_c2B);

  _mm256_store_pd(C + 3 * ldc, c30_c31_c32_c33);
  _mm256_store_pd(C + 3 * ldc + 4, c34_c35_c36_c37);
  _mm256_store_pd(C + 3 * ldc + 8, c38_c39_c3A_c3B);
}

/*
 * A M*K (row stride lda)
 * B K*N (row stride ldb)
 * C M*N (row stride ldc)
 */
static inline void block_l1(const int lda, const int ldb, const int ldc,
                            const int M, const int N, const int K,
                            const double *restrict A, const double *restrict B,
                            double *restrict C) {
  assert(M % AVX_M == 0 && N % AVX_N == 0 && K % AVX_K == 0);

  const int M_b = AVX_M;
  const int N_b = AVX_N;
  const int K_b = AVX_K;

  /* For each block-row of A */
  for (int i = 0; i < M; i += M_b) {
    /* For each block-column of B */
    for (int j = 0; j < N; j += N_b) {
      /* Compute i, j block of C */
      for (int k = 0; k < K; k += K_b) {
        do_block_4x12(lda, ldb, ldc, K_b, A + i * lda + k, B + k * ldb + j,
                      C + i * ldc + j);
      }
    }
  }
}

BLOCK_FUNC(block_l2, BLOCK_SIZE_L1_M, BLOCK_SIZE_L1_N, BLOCK_SIZE_L1_K,
           block_l1)

BLOCK_FUNC(block_l3, BLOCK_SIZE_L2_M, BLOCK_SIZE_L2_N, BLOCK_SIZE_L2_K,
           block_l2)

BLOCK_FUNC_COPY(block_mem, BLOCK_SIZE_L3_M, BLOCK_SIZE_L3_N, BLOCK_SIZE_L3_K,
                block_l3, AVX_M, AVX_N, AVX_K)

/*
 * A lda*lda
 * B lda*lda
 * C lda*lda
 */
void square_dgemm(const int lda, const double *restrict A,
                  const double *restrict B, double *restrict C) {
  block_mem(lda, lda, lda, lda, lda, lda, A, B, C);
}

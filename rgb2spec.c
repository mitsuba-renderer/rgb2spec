#include "rgb2spec.h"
#include "macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define rgb2spec_min(a, b) (((a) < (b)) ? (a) : (b))
#define rgb2spec_max(a, b) (((a) > (b)) ? (a) : (b))

/// Load a RGB2Spec model from disk
RGB2Spec *rgb2spec_load(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f)
        return NULL;

    char header[4];
    if (fread(header, 4, 1, f) != 1 || memcmp(header, "SPEC", 4) != 0) {
        fclose(f);
        return NULL;
    }

    RGB2Spec *m = (RGB2Spec *) malloc(sizeof(RGB2Spec));
    if (!m) {
        fclose(f);
        return NULL;
    }
    m->scale = NULL;
    m->data  = NULL;
    m->fwd   = NULL;
    m->nfine = 0;

    if (fread(&m->res, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        rgb2spec_free(m);
        return NULL;
    }

    size_t size_scale = sizeof(float) * m->res,
           size_data  = sizeof(float) * m->res * m->res *
                        m->res * 3 * RGB2SPEC_N_COEFFS;

    m->scale = (float *) malloc(size_scale);
    m->data = (float *) malloc(size_data);

    if (!m->data || !m->scale ||
        fread(m->scale, size_scale, 1, f) != 1 ||
        fread(m->data, size_data, 1, f) != 1) {
        fclose(f);
        rgb2spec_free(m);
        return NULL;
    }

    // Forward-model block consumed by rgb2spec_fetch_opt().
    if (fread(&m->nfine, sizeof(uint32_t), 1, f) != 1 || m->nfine == 0) {
        fclose(f);
        rgb2spec_free(m);
        return NULL;
    }

    size_t size_fwd = sizeof(float) * (4 * (size_t) m->nfine + 12);
    m->fwd = (float *) malloc(size_fwd);
    if (!m->fwd || fread(m->fwd, size_fwd, 1, f) != 1) {
        fclose(f);
        rgb2spec_free(m);
        return NULL;
    }

    fclose(f);
    return m;
}

/// Release all memory associated with a RGB2Spec model
void rgb2spec_free(RGB2Spec *model) {
    free(model->scale);
    free(model->data);
    free(model->fwd);
    free(model);
}

static int rgb2spec_find_interval(float *values, int size_, float x) {
    int left = 0,
        last_interval = size_ - 2,
        size = last_interval;

    while (size > 0) {
        int half   = size >> 1,
            middle = left + half + 1;

        if (values[middle] <= x) {
            left = middle;
            size -= half + 1;
        } else {
            size = half;
        }
    }

    return rgb2spec_min(left, last_interval);
}

/// Try to use an analytic solution for monochromatic inputs or return 0 (failure)
static int rgb2spec_fetch_mono(const float rgb[3], float out[RGB2SPEC_N_COEFFS]) {
    if (rgb[0] != rgb[1] || rgb[1] != rgb[2])
        return 0;

    // Black/white map to +/- 8192, which evaluate to 0/1 in single precision.
    // (we cannot use infinity as this would produce NaNs)
    float v = rgb[0], r;
    if (v <= 0.f)
        r = -8192.f;
    else if (v >= 1.f)
        r = 8192.f;
    else
        r = (v - .5f) / sqrtf(v * (1.f - v));

    out[0] = out[1] = 0.f;
    out[2] = r;
    return 1;
}

/// Convert an RGB value into a RGB2Spec coefficient representation
void rgb2spec_fetch(RGB2Spec *model, float rgb_[3], float out[RGB2SPEC_N_COEFFS]) {
    // Determine largest RGB component
    int i = 0, res = model->res;
    float rgb[3];
    for (int j = 0; j < 3; ++j)
        rgb[j] = rgb2spec_max(rgb2spec_min(rgb_[j], 1.f), 0.f);

    // Use a closed form solution for monochromatic inputs
    if (rgb2spec_fetch_mono(rgb, out))
        return;

    for (int j = 1; j < 3; ++j)
        if (rgb[j] >= rgb[i])
            i = j;

    float z     = rgb[i],
          scale,
          x,
          y;

    if (z <= 0.f) {
        for (int j = 0; j < RGB2SPEC_N_COEFFS; ++j)
            out[j] = model->data[j];
        return;
    }

    scale = (res - 1) / z;
    x     = rgb[(i + 1) % 3] * scale;
    y     = rgb[(i + 2) % 3] * scale;

    // Trilinearly interpolated lookup
    uint32_t xi = rgb2spec_min((uint32_t) x, (uint32_t) (res - 2)),
             yi = rgb2spec_min((uint32_t) y, (uint32_t) (res - 2)),
             zi = rgb2spec_find_interval(model->scale, model->res, z),
             offset = (((i * res + zi) * res + yi) * res + xi) * RGB2SPEC_N_COEFFS,
             dx = RGB2SPEC_N_COEFFS,
             dy = RGB2SPEC_N_COEFFS * res,
             dz = RGB2SPEC_N_COEFFS * res * res;

    float x1 = x - xi, x0 = 1.f - x1,
          y1 = y - yi, y0 = 1.f - y1,
          z1 = (z - model->scale[zi]) /
               (model->scale[zi + 1] - model->scale[zi]),
          z0 = 1.f - z1;

    for (int j = 0; j < RGB2SPEC_N_COEFFS; ++j) {
        out[j] = ((model->data[offset               ] * x0 +
                   model->data[offset + dx          ] * x1) * y0 +
                  (model->data[offset + dy          ] * x0 +
                   model->data[offset + dy + dx     ] * x1) * y1) * z0 +
                 ((model->data[offset + dz          ] * x0 +
                   model->data[offset + dz + dx     ] * x1) * y0 +
                  (model->data[offset + dz + dy     ] * x0 +
                   model->data[offset + dz + dy + dx] * x1) * y1) * z1;
        offset++;
    }
}

/* ---- One Levenberg-Marquardt refinement step (fetch_opt) ----
 * The problem is parameterized using scaled wavelengths (lambda in [0,1]) so that
 * the math below stays well conditioned in single precision. */

/* Relative tristimulus t = (rgb_to_xyz * rgb) / whitepoint, shared by the two
   CIELab helpers below. */
static void rgb2spec_xyz_rel(const float *r2x, const float *wp,
                             const float rgb[3], float t[3]) {
    for (int k = 0; k < 3; ++k)
        t[k] = (rgb[0]*r2x[k*3] + rgb[1]*r2x[k*3 + 1] + rgb[2]*r2x[k*3 + 2]) / wp[k];
}

static void rgb2spec_lab(const float *r2x, const float *wp,
                         const float rgb[3], float lab[3]) {
    float t[3];
    rgb2spec_xyz_rel(r2x, wp, rgb, t);
    float f[3];
    for (int k = 0; k < 3; ++k) {
        float u = t[k], d = 6.f / 29.f;
        f[k] = (u > d*d*d) ? cbrtf(u) : u / (3.f*d*d) + 4.f / 29.f;
    }
    lab[0] = 116.f * f[1] - 16.f;
    lab[1] = 500.f * (f[0] - f[1]);
    lab[2] = 200.f * (f[1] - f[2]);
}

/* Evaluate the CIELab color of an RGB triple, returning the value in 'lab' and
   its 3x3 Jacobian d(lab)/d(rgb) in 'jac'.
     r2x: row-major 3x3 RGB->XYZ matrix      wp:  XYZ whitepoint (3)
     rgb: input color (3)                    lab: output CIELab (3)
     jac: output Jacobian, jac[a][b] = d lab[a] / d rgb[b]  */
static void rgb2spec_lab_jac(const float *r2x, const float *wp,
                             const float rgb[3], float lab[3], float jac[3][3]) {
    float t[3];
    rgb2spec_xyz_rel(r2x, wp, rgb, t);
    float f[3], g[3];
    for (int k = 0; k < 3; ++k) {
        float u = t[k], d = 6.f / 29.f;
        if (u > d*d*d) {
            float cr = cbrtf(u);
            f[k] = cr;
            g[k] = (1.f/3.f) / (cr*cr * wp[k]);
        } else {
            f[k] = u / (3.f*d*d) + 4.f / 29.f;
            g[k] = 1.f / (3.f*d*d * wp[k]);
        }
    }
    lab[0] = 116.f * f[1] - 16.f;
    lab[1] = 500.f * (f[0] - f[1]);
    lab[2] = 200.f * (f[1] - f[2]);
    float G[3][3] = {
        {        0.f,  116.f * g[1],          0.f },
        { 500.f*g[0], -500.f * g[1],          0.f },
        {        0.f,  200.f * g[1], -200.f * g[2] }
    };
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            float s = 0;
            for (int k = 0; k < 3; ++k)
                s += G[a][k] * r2x[k*3 + b];
            jac[a][b] = s;
        }
}

/* Solve 3x3 A x = b (Gaussian elimination, partial pivot). Returns 0 if singular. */
static int rgb2spec_solve3(float A[3][3], const float b[3], float x[3]) {
    float M[3][4] = { { A[0][0], A[0][1], A[0][2], b[0] },
                      { A[1][0], A[1][1], A[1][2], b[1] },
                      { A[2][0], A[2][1], A[2][2], b[2] } };
    for (int c = 0; c < 3; ++c) {
        int p = c;
        for (int r = c + 1; r < 3; ++r)
            if (fabsf(M[r][c]) > fabsf(M[p][c])) p = r;
        if (fabsf(M[p][c]) < 1e-30f) return 0;
        for (int k = 0; k < 4; ++k) { float tmp = M[c][k]; M[c][k] = M[p][k]; M[p][k] = tmp; }
        for (int r = 0; r < 3; ++r)
            if (r != c) {
                float fct = M[r][c] / M[c][c];
                for (int k = c; k < 4; ++k) M[r][k] -= fct * M[c][k];
            }
    }
    for (int r = 0; r < 3; ++r) x[r] = M[r][3] / M[r][r];
    return 1;
}

/* Relax FP / enable vectorization for the integration loops below (see macros.h). */
RGB2SPEC_FP_PUSH

/* Squared CIELab residual of a candidate coefficient vector 'c' (re-integrates
   the sigmoid spectrum). Used to accept or reject a trial step. */
RGB2SPEC_FP_FAST
static float rgb2spec_opt_cost(int n, const float *lam, const float *W,
                               const float *r2x, const float *wp,
                               const float tgt_lab[3], const float c[3]) {
    RGB2SPEC_FP_REASSOC
    float o[3] = { 0, 0, 0 };
    for (int i = 0; i < n; ++i) {
        float l = lam[i], P = (c[0]*l + c[1])*l + c[2];
        float q = 1.f / sqrtf(1.f + P*P), s = .5f*P*q + .5f;
        for (int j = 0; j < 3; ++j)
            o[j] += W[j*n + i] * s;
    }
    float olab[3];
    rgb2spec_lab(r2x, wp, o, olab);
    float d0 = tgt_lab[0]-olab[0], d1 = tgt_lab[1]-olab[1], d2 = tgt_lab[2]-olab[2];
    return d0*d0 + d1*d1 + d2*d2;
}

RGB2SPEC_FP_FAST
void rgb2spec_fetch_opt(RGB2Spec *model, float rgb[3], float out[RGB2SPEC_N_COEFFS]) {
    RGB2SPEC_FP_REASSOC

    float tgt[3];
    for (int j = 0; j < 3; ++j)
        tgt[j] = rgb2spec_max(rgb2spec_min(rgb[j], 1.f), 0.f);

    // Use a closed form solution for monochromatic inputs
    if (rgb2spec_fetch_mono(tgt, out))
        return;

    rgb2spec_fetch(model, rgb, out);

    const float c0 = 360.f, c1 = 1.f / (830.f - 360.f);

    // Interpolated coeffs are wavelength-domain; move to the scaled domain.
    float A = out[0], B = out[1], C = out[2];
    float c[3] = { A / (c1*c1),
                   2.f*A*c0/c1 + B/c1,
                   A*c0*c0 + B*c0 + C };

    int n = (int) model->nfine;
    const float *lam = model->fwd;
    const float *W   = model->fwd + n;          // W[j*n + i] = rgb_tbl[j][i]
    const float *r2x = model->fwd + 4*n;
    const float *wp  = model->fwd + 4*n + 9;

    // Single pass: reproduced color and its Jacobian w.r.t. the coefficients.
    float orgb[3] = { 0, 0, 0 };
    float dout[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    for (int i = 0; i < n; ++i) {
        float l = lam[i], P = (c[0]*l + c[1])*l + c[2];
        float q = 1.f / sqrtf(1.f + P*P), s = .5f*P*q + .5f, sp = .5f*q*q*q;
        float dP0 = l*l, dP1 = l;
        for (int j = 0; j < 3; ++j) {
            float w = W[j*n + i];
            orgb[j] += w * s;
            float wsp = w * sp;
            dout[j][0] += wsp * dP0;
            dout[j][1] += wsp * dP1;
            dout[j][2] += wsp;
        }
    }

    float tgt_lab[3], res[3], olab[3], lab_jac[3][3], J[3][3];
    rgb2spec_lab(r2x, wp, tgt, tgt_lab);
    rgb2spec_lab_jac(r2x, wp, orgb, olab, lab_jac);   // Lab + Jacobian, one pass
    for (int j = 0; j < 3; ++j) res[j] = tgt_lab[j] - olab[j];
    float r0 = res[0]*res[0] + res[1]*res[1] + res[2]*res[2];

    // J = d(residual)/d(coeffs); then the normal-equation terms J^T J and J^T res.
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            float s = 0;
            for (int k = 0; k < 3; ++k)
                s += lab_jac[a][k] * dout[k][b];
            J[a][b] = -s;
        }
    float JtJ[3][3], g[3] = { 0, 0, 0 };
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            float s = 0;
            for (int k = 0; k < 3; ++k)
                s += J[k][a] * J[k][b];
            JtJ[a][b] = s;
        }
        for (int k = 0; k < 3; ++k)
            g[a] += J[k][a] * res[k];
    }

    // One Levenberg-Marquardt step: solve (J^T J + lambda*I) step = g, coeffs -=
    // step. lambda = 0 is the undamped Gauss-Newton step, near-exact in the gamut
    // interior. Near the boundary the spectrum saturates and that step overshoots,
    // so raise 'lambda' (shortening the step toward gradient descent) until one
    // reduces the residual below the un-refined r0.
    float lambda = 0.f;
    for (int t = 0; t < 20; ++t) {
        float M[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                M[a][b] = JtJ[a][b] + (a == b ? lambda : 0.f);
        float step[3];
        if (rgb2spec_solve3(M, g, step)) {
            float trial[3] = { c[0]-step[0], c[1]-step[1], c[2]-step[2] };
            if (rgb2spec_opt_cost(n, lam, W, r2x, wp, tgt_lab, trial) < r0) {
                c[0] = trial[0]; c[1] = trial[1]; c[2] = trial[2];
                break;
            }
        }
        lambda = lambda ? lambda * 10.f : 1e-3f;
    }

    // Back to the wavelength domain (the format rgb2spec_eval expects).
    out[0] = c[0] * (c1*c1);
    out[1] = c[1]*c1 - 2.f*c[0]*c0*(c1*c1);
    out[2] = c[2] - c[1]*c0*c1 + c[0]*(c0*c1)*(c0*c1);
}
RGB2SPEC_FP_POP

static inline float rgb2spec_fma(float a, float b, float c) {
    #if defined(__FMA__)
        // Only use fmaf() if implemented in hardware
        return fmaf(a, b, c);
    #else
        return a*b + c;
    #endif
}

float rgb2spec_eval_precise(float coeff[RGB2SPEC_N_COEFFS], float lambda) {
    float x = rgb2spec_fma(rgb2spec_fma(coeff[0], lambda, coeff[1]), lambda, coeff[2]),
          y = 1.f / sqrtf(rgb2spec_fma(x, x, 1.f));
    return rgb2spec_fma(.5f * x, y, .5f);
}

float rgb2spec_eval_fast(float coeff[RGB2SPEC_N_COEFFS], float lambda) {
    float x = rgb2spec_fma(rgb2spec_fma(coeff[0], lambda, coeff[1]), lambda, coeff[2]),
#if RGB2SPEC_HAS_X86_INTRINSICS && (defined(__SSE__) || defined(__x86_64__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1))
          y = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(rgb2spec_fma(x, x, 1.f))));
#else
          y = 1.f / sqrtf(rgb2spec_fma(x, x, 1.f));
#endif
    return rgb2spec_fma(.5f * x, y, .5f);
}

#if RGB2SPEC_HAS_X86_INTRINSICS && defined(__SSE4_2__)
static inline __m128 rgb2spec_fma128(__m128 a, __m128 b, __m128 c) {
    #if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
    #else
        /// Fallback for pre-Haswell architectures
        return _mm_add_ps(_mm_mul_ps(a, b), c);
    #endif
}

__m128 rgb2spec_eval_sse(float coeff[RGB2SPEC_N_COEFFS], __m128 lambda) {
    __m128 c0 = _mm_set1_ps(coeff[0]), c1 = _mm_set1_ps(coeff[1]),
           c2 = _mm_set1_ps(coeff[2]), h = _mm_set1_ps(.5f),
           o = _mm_set1_ps(1.f);

    __m128 x = rgb2spec_fma128(rgb2spec_fma128(c0, lambda, c1), lambda, c2),
           y = _mm_rsqrt_ps(rgb2spec_fma128(x, x, o));

    return rgb2spec_fma128(_mm_mul_ps(h, x), y, h);
}
#endif

#if RGB2SPEC_HAS_X86_INTRINSICS && defined(__AVX__)
__m256 rgb2spec_fma256(__m256 a, __m256 b, __m256 c) {
    #if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
    #else
        /// Fallback for pre-Haswell architectures
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
    #endif
}

__m256 rgb2spec_eval_avx(float coeff[RGB2SPEC_N_COEFFS], __m256 lambda) {
    __m256 c0 = _mm256_set1_ps(coeff[0]), c1 = _mm256_set1_ps(coeff[1]),
           c2 = _mm256_set1_ps(coeff[2]), h = _mm256_set1_ps(.5f),
           o = _mm256_set1_ps(1.f);

    __m256 x = rgb2spec_fma256(rgb2spec_fma256(c0, lambda, c1), lambda, c2),
           y = _mm256_rsqrt_ps(rgb2spec_fma256(x, x, o));

    return rgb2spec_fma256(_mm256_mul_ps(h, x), y, h);
}
#endif

#if RGB2SPEC_HAS_X86_INTRINSICS && defined(__AVX512F__)
__m512 rgb2spec_eval_avx512(float coeff[RGB2SPEC_N_COEFFS], __m512 lambda) {
    __m512 c0 = _mm512_set1_ps(coeff[0]), c1 = _mm512_set1_ps(coeff[1]),
           c2 = _mm512_set1_ps(coeff[2]), h = _mm512_set1_ps(.5f),
           o = _mm512_set1_ps(1.f);

    __m512 x = _mm512_fmadd_ps(_mm512_fmadd_ps(c0, lambda, c1), lambda, c2),
           y = _mm512_rsqrt14_ps(_mm512_fmadd_ps(x, x, o));

    return _mm512_fmadd_ps(_mm512_mul_ps(h, x), y, h);
}
#endif

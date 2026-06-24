#if defined(_MSC_VER)
#  define NOMINMAX
#  define strcasecmp _stricmp
#endif

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#include <thread>
#include <atomic>
#include <vector>

#include "macros.h"

/* Working precision for the optimizer. Single precision is sufficient for the
 * table's accuracy and lets the integration loops auto-vectorize to 4-wide SIMD
 * (define this to `double` to fall back to double precision). The reference
 * colorimetric data in cie1931.h deliberately stays double. */
#define Float float

#include "details/cie1931.h"
#include "details/lu.h"

/// Discretization of quadrature scheme
#define CIE_FINE_SAMPLES ((CIE_SAMPLES - 1) * 3 + 1)

/// Sample count padded to a multiple of 16 (the widest SIMD lane count for
/// float) so the quadrature loop vectorizes with no scalar remainder. The extra
/// entries stay zero and therefore contribute nothing to the integral.
#define CIE_FINE_SAMPLES_PAD (((CIE_FINE_SAMPLES + 15) / 16) * 16)

/// Precomputed tables for fast spectral -> RGB conversion. 64-byte aligned (and
/// each rgb_tbl row is a multiple of 64 bytes) for aligned AVX-512 loads.
alignas(64) Float lambda_tbl[CIE_FINE_SAMPLES_PAD];
alignas(64) Float rgb_tbl[3][CIE_FINE_SAMPLES_PAD];
Float rgb_to_xyz[3][3],
      xyz_to_rgb[3][3],
      xyz_whitepoint[3];

/// Currently supported gamuts
enum Gamut {
    SRGB,
    ProPhotoRGB,
    ACES2065_1,
    REC2020,
    ERGB,
    XYZ,
    NO_GAMUT,
};

Float sigmoid(Float x) {
    return Float(0.5) * x / std::sqrt(Float(1) + x * x) + Float(0.5);
}

Float smoothstep(Float x) {
    return x * x * (Float(3) - Float(2) * x);
}

Float sqr(Float x) { return x * x; }

void cie_lab(Float *p) {
    Float X = 0, Y = 0, Z = 0,
      Xw = xyz_whitepoint[0],
      Yw = xyz_whitepoint[1],
      Zw = xyz_whitepoint[2];

    for (int j = 0; j < 3; ++j) {
        X += p[j] * rgb_to_xyz[0][j];
        Y += p[j] * rgb_to_xyz[1][j];
        Z += p[j] * rgb_to_xyz[2][j];
    }

    auto f = [](Float t) -> Float {
        Float delta = Float(6.0 / 29.0);
        if (t > delta*delta*delta)
            return std::cbrt(t);
        else
            return t / (delta*delta * Float(3)) + Float(4.0 / 29.0);
    };

    p[0] = Float(116) * f(Y / Yw) - Float(16);
    p[1] = Float(500) * (f(X / Xw) - f(Y / Yw));
    p[2] = Float(200) * (f(Y / Yw) - f(Z / Zw));
}

/**
 * This function precomputes tables used to convert arbitrary spectra
 * to RGB (either sRGB or ProPhoto RGB)
 *
 * A composite quadrature rule integrates the CIE curves, reflectance, and
 * illuminant spectrum over each 5nm segment in the 360..830nm range using
 * Simpson's 3/8 rule (4th-order accurate), which evaluates the integrand at
 * four positions per segment. While the CIE curves and illuminant spectrum are
 * linear over the segment, the reflectance could have arbitrary behavior,
 * hence the extra precations.
 *
 * The accumulation is done in double precision (reference data is double) and
 * the results are stored into the single-precision working tables.
 */
void init_tables(Gamut gamut) {
    memset(rgb_tbl, 0, sizeof(rgb_tbl));

    double h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (CIE_FINE_SAMPLES - 1);

    const double *illuminant = nullptr;
    const double (*to_rgb)[3] = nullptr, (*to_xyz)[3] = nullptr;

    switch (gamut) {
        case SRGB:        illuminant = cie_d65; to_rgb = xyz_to_srgb;          to_xyz = srgb_to_xyz;          break;
        case ERGB:        illuminant = cie_e;   to_rgb = xyz_to_ergb;          to_xyz = ergb_to_xyz;          break;
        case XYZ:         illuminant = cie_e;   to_rgb = xyz_to_xyz;           to_xyz = xyz_to_xyz;           break;
        case ProPhotoRGB: illuminant = cie_d50; to_rgb = xyz_to_prophoto_rgb;  to_xyz = prophoto_rgb_to_xyz;  break;
        case ACES2065_1:  illuminant = cie_d60; to_rgb = xyz_to_aces2065_1;    to_xyz = aces2065_1_to_xyz;    break;
        case REC2020:     illuminant = cie_d65; to_rgb = xyz_to_rec2020;       to_xyz = rec2020_to_xyz;       break;
        default: throw std::runtime_error("init_gamut(): invalid/unsupported gamut.");
    }

    /* Reference matrices are double; convert element-wise into the Float tables */
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            xyz_to_rgb[a][b] = (Float) to_rgb[a][b];
            rgb_to_xyz[a][b] = (Float) to_xyz[a][b];
        }

    double whitepoint[3] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
        double lambda = CIE_LAMBDA_MIN + i * h;

        double xyz[3] = { cie_interp(cie_x, lambda),
                          cie_interp(cie_y, lambda),
                          cie_interp(cie_z, lambda) },
               I = cie_interp(illuminant, lambda);

        double weight = 3.0 / 8.0 * h;
        if (i == 0 || i == CIE_FINE_SAMPLES - 1)
            ;
        else if ((i - 1) % 3 == 2)
            weight *= 2.0;
        else
            weight *= 3.0;

        lambda_tbl[i] = (Float) lambda;
        for (int k = 0; k < 3; ++k) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j)
                acc += to_rgb[k][j] * xyz[j] * I * weight;
            rgb_tbl[k][i] = (Float) acc;
        }

        for (int channel = 0; channel < 3; ++channel)
            whitepoint[channel] += xyz[channel] * I * weight;
    }

    for (int channel = 0; channel < 3; ++channel)
        xyz_whitepoint[channel] = (Float) whitepoint[channel];
}

/* Relax FP / enable vectorization for the integration loops below (see macros.h). */
RGB2SPEC_FP_PUSH

RGB2SPEC_FP_FAST
void eval_residual(const Float *coeffs, const Float *rgb, Float *residual) {
    RGB2SPEC_FP_REASSOC
    Float out[3] = { 0, 0, 0 };
    const Float lo  = (Float) CIE_LAMBDA_MIN,
                inv = (Float) (1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN));
    Float c0 = coeffs[0], c1 = coeffs[1], c2 = coeffs[2];

    for (int i = 0; i < CIE_FINE_SAMPLES_PAD; ++i) {
        Float lambda = (lambda_tbl[i] - lo) * inv;          /* scale to 0..1 */
        Float x = (c0 * lambda + c1) * lambda + c2;         /* polynomial */
        Float s = Float(0.5) * x / std::sqrt(Float(1) + x * x) + Float(0.5);

        for (int j = 0; j < 3; ++j)
            out[j] += rgb_tbl[j][i] * s;
    }
    cie_lab(out);
    for (int j = 0; j < 3; ++j) residual[j] = rgb[j];
    cie_lab(residual);

    for (int j = 0; j < 3; ++j)
        residual[j] -= out[j];
}

/// Evaluate the CIELab color of the RGB input 'in', returning the value in 'lab'
/// and its analytic Jacobian d(lab)/d(in) in 'jac'.
void cie_lab_jac(const Float in[3], Float lab[3], Float jac[3][3]) {
    Float wn[3] = { xyz_whitepoint[0], xyz_whitepoint[1], xyz_whitepoint[2] };

    Float xyz[3] = { 0, 0, 0 };
    for (int a = 0; a < 3; ++a)
        for (int j = 0; j < 3; ++j)
            xyz[a] += in[j] * rgb_to_xyz[a][j];

    Float fv[3], gd[3];      /* f(t) and f'(t)/w_n, with t = xyz / whitepoint */
    for (int k = 0; k < 3; ++k) {
        Float t = xyz[k] / wn[k], delta = Float(6.0 / 29.0);
        if (t > delta*delta*delta) {
            Float cr = std::cbrt(t);
            fv[k] = cr;
            gd[k] = Float(1.0 / 3.0) / (cr*cr * wn[k]);
        } else {
            fv[k] = t / (delta*delta * Float(3)) + Float(4.0 / 29.0);
            gd[k] = Float(1) / (delta*delta * Float(3) * wn[k]);
        }
    }

    lab[0] = Float(116) * fv[1] - Float(16);
    lab[1] = Float(500) * (fv[0] - fv[1]);
    lab[2] = Float(200) * (fv[1] - fv[2]);

    /* d Lab / d XYZ */
    Float g[3][3] = {
        {            0,   Float(116) * gd[1],                0 },
        { Float(500)*gd[0],  Float(-500) * gd[1],            0 },
        {            0,   Float(200) * gd[1],  Float(-200)*gd[2] }
    };

    /* d Lab / d RGB = (d Lab / d XYZ) * rgb_to_xyz */
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            Float s = 0;
            for (int k = 0; k < 3; ++k)
                s += g[a][k] * rgb_to_xyz[k][b];
            jac[a][b] = s;
        }
}

/// Evaluate the residual together with its Jacobian in a single integration pass.
RGB2SPEC_FP_FAST
void eval_residual_jac(const Float *coeffs, const Float *rgb,
                       Float *residual, Float jac[3][3]) {
    RGB2SPEC_FP_REASSOC
    Float out[3] = { 0, 0, 0 };
    Float dout[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }; /* d out[j] / d coeffs[a] */
    const Float lo  = (Float) CIE_LAMBDA_MIN,
                inv = (Float) (1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN));
    Float c0 = coeffs[0], c1 = coeffs[1], c2 = coeffs[2];

    for (int i = 0; i < CIE_FINE_SAMPLES_PAD; ++i) {
        Float lambda = (lambda_tbl[i] - lo) * inv;
        Float x = (c0 * lambda + c1) * lambda + c2;
        Float dx0 = lambda * lambda, dx1 = lambda;            /* dP/dcoeffs */

        /* Sigmoid and its derivative (both share q = 1 / sqrt(1 + x^2)) */
        Float q  = Float(1) / std::sqrt(Float(1) + x * x);
        Float s  = Float(0.5) * x * q + Float(0.5);
        Float sp = Float(0.5) * q * q * q;

        for (int j = 0; j < 3; ++j) {
            Float w = rgb_tbl[j][i];
            out[j] += w * s;
            Float wsp = w * sp;
            dout[j][0] += wsp * dx0;
            dout[j][1] += wsp * dx1;
            dout[j][2] += wsp;
        }
    }

    /* Residual in CIELab. The reproduced color needs both its Lab value and the
       Lab Jacobian, so compute them together in one tristimulus pass. */
    Float out_lab[3], lab_jac[3][3];
    cie_lab_jac(out, out_lab, lab_jac);
    for (int j = 0; j < 3; ++j) residual[j] = rgb[j];
    cie_lab(residual);
    for (int j = 0; j < 3; ++j)
        residual[j] -= out_lab[j];

    /* Chain rule: d residual / d coeffs = -(d Lab / d out) * (d out / d coeffs) */
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            Float s = 0;
            for (int k = 0; k < 3; ++k)
                s += lab_jac[a][k] * dout[k][b];
            jac[a][b] = -s;
        }
}
RGB2SPEC_FP_POP

/**
 * Find the polynomial coefficients whose sigmoidal spectrum best reproduces the
 * target color 'rgb'. The objective is the squared CIELab distance, minimized
 * with the Levenberg-Marquardt algorithm.
 *
 * Inside the gamut the target color is exactly reachable, the Jacobian is well
 * conditioned, and LM converges quadratically to a zero residual. Near or beyond
 * the gamut boundary no exact solution exists and the Jacobian becomes singular
 * as the spectrum saturates; there the adaptive 'lambda' damping regularizes the
 * step so the method settles at the closest achievable color instead of diverging.
 */
Float LM(const Float rgb[3], Float coeffs[3], int it = 15) {
    Float residual[3], jac[3][3];
    eval_residual_jac(coeffs, rgb, residual, jac);
    Float cost = sqr(residual[0]) + sqr(residual[1]) + sqr(residual[2]);

    Float lambda = Float(1e-3);

    for (int i = 0; i < it && cost > Float(1e-12); ++i) {
        /* Assemble the normal equations: A = J^T J,  g = J^T residual */
        Float A[3][3], g[3] = { 0, 0, 0 };
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                Float s = 0;
                for (int k = 0; k < 3; ++k)
                    s += jac[k][a] * jac[k][b];
                A[a][b] = s;
            }
            for (int k = 0; k < 3; ++k)
                g[a] += jac[k][a] * residual[k];
        }

        /* Try damped steps, increasing 'lambda', until one decreases the cost */
        bool accepted = false;
        for (int t = 0; t < 10 && !accepted; ++t) {
            Float M0[3], M1[3], M2[3], *M[3] = { M0, M1, M2 };
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    M[a][b] = A[a][b] + (a == b ? lambda : Float(0)); /* A + lambda*I */

            int P[4];
            if (LUPDecompose(M, 3, Float(1e-30), P) != 1) {
                lambda *= Float(10); /* singular even when damped -> damp harder */
                continue;
            }

            /* Solve (A + lambda*I) step = g;  the LM update is coeffs -= step.
               Trials only need the cost, so use the cheaper residual-only eval. */
            Float step[3];
            LUPSolve(M, P, g, 3, step);

            Float trial[3] = { coeffs[0] - step[0],
                               coeffs[1] - step[1],
                               coeffs[2] - step[2] };
            Float trial_res[3];
            eval_residual(trial, rgb, trial_res);
            Float trial_cost = sqr(trial_res[0]) + sqr(trial_res[1]) +
                               sqr(trial_res[2]);

            if (trial_cost < cost) {
                memcpy(coeffs, trial, sizeof(Float) * 3);
                cost = trial_cost;
                lambda = std::max(lambda * Float(0.5), Float(1e-12)); /* step worked: trust the model more */
                accepted = true;
            } else {
                lambda *= Float(10); /* step failed: trust the model less */
                if (lambda > Float(1e12))
                    break;
            }
        }

        if (!accepted)
            break; /* converged, or no damped step can improve further */

        /* Refresh residual and analytic Jacobian at the accepted point */
        eval_residual_jac(coeffs, rgb, residual, jac);
    }

    return std::sqrt(cost);
}

static Gamut parse_gamut(const char *str) {
    if (!strcasecmp(str, "sRGB"))
        return SRGB;
    if (!strcasecmp(str, "eRGB"))
        return ERGB;
    if (!strcasecmp(str, "XYZ"))
        return XYZ;
    if (!strcasecmp(str, "ProPhotoRGB"))
        return ProPhotoRGB;
    if (!strcasecmp(str, "ACES2065_1"))
        return ACES2065_1;
    if (!strcasecmp(str, "REC2020"))
        return REC2020;
    return NO_GAMUT;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Syntax: rgb2spec_opt <resolution> <output> [<gamut>]\n"
               "where <gamut> is one of sRGB,eRGB,XYZ,ProPhotoRGB,ACES2065_1,REC2020\n");
        exit(-1);
    }
    Gamut gamut = SRGB;
    if (argc > 3) gamut = parse_gamut(argv[3]);
    if (gamut == NO_GAMUT) {
        fprintf(stderr, "Could not parse gamut `%s'!\n", argv[3]);
        exit(-1);
    }
    init_tables(gamut);

    const int res = atoi(argv[1]);
    if (res == 0) {
        printf("Invalid resolution!\n");
        exit(-1);
    }

    printf("Optimizing spectra ");

    float *scale = new float[res];
    for (int k = 0; k < res; ++k)
        scale[k] = (float) smoothstep(smoothstep(Float(k) / Float(res - 1)));

    size_t bufsize = 3*3*res*res*res;
    float *out = new float[bufsize];

    /* Each (l, j) slice is an independent unit of work: distinct slices write to
       disjoint regions of 'out', so the tasks need no synchronization. */
    auto process = [&](int l, int j) {
        const Float y = Float(j) / Float(res - 1);
        printf(".");
        fflush(stdout);
        for (int i = 0; i < res; ++i) {
            const Float x = Float(i) / Float(res - 1);
            Float coeffs[3], rgb[3];
            memset(coeffs, 0, sizeof(Float)*3);

            int start = res / 5;

            for (int k = start; k < res; ++k) {
                Float b = scale[k];

                rgb[l] = b;
                rgb[(l + 1) % 3] = x*b;
                rgb[(l + 2) % 3] = y*b;

                Float resid = LM(rgb, coeffs);
                (void) resid;

                /* Remap polynomial from the [0,1] domain back to wavelength (nm);
                   kept in double precision since it is one store per node. */
                double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                double A = coeffs[0], B = coeffs[1], C = coeffs[2];

                int idx = ((l*res + k) * res + j)*res+i;

                out[3*idx + 0] = float(A*(c1*c1));
                out[3*idx + 1] = float(B*c1 - 2*A*c0*(c1*c1));
                out[3*idx + 2] = float(C - B*c0*c1 + A*(c0*c1)*(c0*c1));
                //out[3*idx + 2] = resid;
            }

            memset(coeffs, 0, sizeof(Float)*3);
            for (int k = start; k>=0; --k) {
                Float b = scale[k];

                rgb[l] = b;
                rgb[(l + 1) % 3] = x*b;
                rgb[(l + 2) % 3] = y*b;

                Float resid = LM(rgb, coeffs);
                (void) resid;

                double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                double A = coeffs[0], B = coeffs[1], C = coeffs[2];

                int idx = ((l*res + k) * res + j)*res+i;

                out[3*idx + 0] = float(A*(c1*c1));
                out[3*idx + 1] = float(B*c1 - 2*A*c0*(c1*c1));
                out[3*idx + 2] = float(C - B*c0*c1 + A*(c0*c1)*(c0*c1));
                //out[3*idx + 2] = resid;
            }
        }
    };

    /* A pool of worker threads pulls (l, j) slices from a shared
       atomic counter. This helps because the cost is quite uneven. */
    const int n_tasks = 3 * res;
    std::atomic<int> next_task(0);
    auto worker = [&]() {
        int t;
        while ((t = next_task.fetch_add(1)) < n_tasks)
            process(t / res, t % res);
    };

    unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n_threads; ++t)
        pool.emplace_back(worker);
    for (std::thread &th : pool)
        th.join();

    FILE *f = fopen(argv[2], "wb");
    if (f == nullptr)
        throw std::runtime_error("Could not create file!");
    fwrite("SPEC", 4, 1, f);
    uint32_t resolution = res;
    fwrite(&resolution, sizeof(uint32_t), 1, f);
    fwrite(scale, res * sizeof(float), 1, f);

    fwrite(out, sizeof(float)*bufsize, 1, f);

    /* Append the forward-model block consumed by rgb2spec_fetch_opt():
       nfine, then [lambda: nfine][rgb_tbl: 3*nfine][rgb_to_xyz: 9][whitepoint: 3]. */
    {
        uint32_t nfine = CIE_FINE_SAMPLES_PAD;
        const double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
        float *blk = new float[4*nfine + 12];
        for (uint32_t i = 0; i < nfine; ++i)
            blk[i] = float((lambda_tbl[i] - c0) * c1);
        for (uint32_t i = 0; i < 3*nfine; ++i)
            blk[nfine + i] = ((const float *) rgb_tbl)[i];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                blk[4*nfine + a*3 + b] = (float) rgb_to_xyz[a][b];
        for (int k = 0; k < 3; ++k)
            blk[4*nfine + 9 + k] = (float) xyz_whitepoint[k];
        fwrite(&nfine, sizeof(uint32_t), 1, f);
        fwrite(blk, sizeof(float)*(4*nfine + 12), 1, f);
        delete[] blk;
    }

    delete[] out;
    delete[] scale;
    fclose(f);
    printf(" done.\n");
}

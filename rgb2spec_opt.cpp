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

#include "details/cie1931.h"
#include "details/lu.h"

/// Discretization of quadrature scheme
#define CIE_FINE_SAMPLES ((CIE_SAMPLES - 1) * 3 + 1)

/// Precomputed tables for fast spectral -> RGB conversion
double lambda_tbl[CIE_FINE_SAMPLES],
       rgb_tbl[3][CIE_FINE_SAMPLES],
       rgb_to_xyz[3][3],
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

double sigmoid(double x) {
    return 0.5 * x / std::sqrt(1.0 + x * x) + 0.5;
}

double smoothstep(double x) {
    return x * x * (3.0 - 2.0 * x);
}

double sqr(double x) { return x * x; }

void cie_lab(double *p) {
    double X = 0.0, Y = 0.0, Z = 0.0,
      Xw = xyz_whitepoint[0],
      Yw = xyz_whitepoint[1],
      Zw = xyz_whitepoint[2];

    for (int j = 0; j < 3; ++j) {
        X += p[j] * rgb_to_xyz[0][j];
        Y += p[j] * rgb_to_xyz[1][j];
        Z += p[j] * rgb_to_xyz[2][j];
    }

    auto f = [](double t) -> double {
        double delta = 6.0 / 29.0;
        if (t > delta*delta*delta)
            return cbrt(t);
        else
            return t / (delta*delta * 3.0) + (4.0 / 29.0);
    };

    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
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
 */
void init_tables(Gamut gamut) {
    memset(rgb_tbl, 0, sizeof(rgb_tbl));
    memset(xyz_whitepoint, 0, sizeof(xyz_whitepoint));

    double h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (CIE_FINE_SAMPLES - 1);

    const double *illuminant = nullptr;

    switch (gamut) {
        case SRGB:
            illuminant = cie_d65;
            memcpy(xyz_to_rgb, xyz_to_srgb, sizeof(double) * 9);
            memcpy(rgb_to_xyz, srgb_to_xyz, sizeof(double) * 9);
            break;

        case ERGB:
            illuminant = cie_e;
            memcpy(xyz_to_rgb, xyz_to_ergb, sizeof(double) * 9);
            memcpy(rgb_to_xyz, ergb_to_xyz, sizeof(double) * 9);
            break;

        case XYZ:
            illuminant = cie_e;
            memcpy(xyz_to_rgb, xyz_to_xyz, sizeof(double) * 9);
            memcpy(rgb_to_xyz, xyz_to_xyz, sizeof(double) * 9);
            break;

        case ProPhotoRGB:
            illuminant = cie_d50;
            memcpy(xyz_to_rgb, xyz_to_prophoto_rgb, sizeof(double) * 9);
            memcpy(rgb_to_xyz, prophoto_rgb_to_xyz, sizeof(double) * 9);
            break;

        case ACES2065_1:
            illuminant = cie_d60;
            memcpy(xyz_to_rgb, xyz_to_aces2065_1, sizeof(double) * 9);
            memcpy(rgb_to_xyz, aces2065_1_to_xyz, sizeof(double) * 9);
            break;

        case REC2020:
            illuminant = cie_d65;
            memcpy(xyz_to_rgb, xyz_to_rec2020, sizeof(double) * 9);
            memcpy(rgb_to_xyz, rec2020_to_xyz, sizeof(double) * 9);
            break;

        default:
            throw std::runtime_error("init_gamut(): invalid/unsupported gamut.");
    }

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
            weight *= 2.f;
        else
            weight *= 3.f;

        lambda_tbl[i] = lambda;
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                rgb_tbl[k][i] += xyz_to_rgb[k][j] * xyz[j] * I * weight;

        for (int channel = 0; channel < 3; ++channel)
            xyz_whitepoint[channel] += xyz[channel] * I * weight;
    }
}

void eval_residual(const double *coeffs, const double *rgb, double *residual) {
    double out[3] = { 0.0, 0.0, 0.0 };

    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
        /* Scale lambda to 0..1 range */
        double lambda = (lambda_tbl[i] - CIE_LAMBDA_MIN) /
                        (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

        /* Polynomial */
        double x = 0.0;
        for (int coeff = 0; coeff < 3; ++coeff)
            x = x * lambda + coeffs[coeff];

        /* Sigmoid */
        double s = sigmoid(x);

        /* Integrate against precomputed curves */
        for (int j = 0; j < 3; ++j)
            out[j] += rgb_tbl[j][i] * s;
    }
    cie_lab(out);
    memcpy(residual, rgb, sizeof(double) * 3);
    cie_lab(residual);

    for (int j = 0; j < 3; ++j)
        residual[j] -= out[j];
}

/// Analytic Jacobian of cie_lab() with respect to its RGB input, evaluated at 'rgb'.
void cie_lab_jac(const double rgb[3], double jac[3][3]) {
    double Xw = xyz_whitepoint[0],
           Yw = xyz_whitepoint[1],
           Zw = xyz_whitepoint[2];

    double X = 0.0, Y = 0.0, Z = 0.0;
    for (int j = 0; j < 3; ++j) {
        X += rgb[j] * rgb_to_xyz[0][j];
        Y += rgb[j] * rgb_to_xyz[1][j];
        Z += rgb[j] * rgb_to_xyz[2][j];
    }

    auto fp = [](double t) -> double {
        double delta = 6.0 / 29.0;
        if (t > delta*delta*delta)
            return (1.0 / 3.0) * std::pow(t, -2.0 / 3.0);
        else
            return 1.0 / (delta*delta * 3.0);
    };

    double gx = fp(X / Xw) / Xw,
           gy = fp(Y / Yw) / Yw,
           gz = fp(Z / Zw) / Zw;

    /* d Lab / d XYZ */
    double g[3][3] = {
        {      0.0,   116.0 * gy,        0.0 },
        { 500.0*gx,  -500.0 * gy,        0.0 },
        {      0.0,   200.0 * gy,  -200.0*gz }
    };

    /* d Lab / d RGB = (d Lab / d XYZ) * rgb_to_xyz */
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
                s += g[a][k] * rgb_to_xyz[k][b];
            jac[a][b] = s;
        }
}

/// Evaluate the residual together with its Jacobian in a single integration pass.
void eval_residual_jac(const double *coeffs, const double *rgb,
                       double *residual, double jac[3][3]) {
    double out[3] = { 0.0, 0.0, 0.0 };
    double dout[3][3] = { { 0.0, 0.0, 0.0 },
                          { 0.0, 0.0, 0.0 },
                          { 0.0, 0.0, 0.0 } }; /* d out[j] / d coeffs[a] */

    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
        /* Scale lambda to 0..1 range */
        double lambda = (lambda_tbl[i] - CIE_LAMBDA_MIN) /
                        (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

        /* Polynomial and its derivatives w.r.t. the coefficients */
        double x = (coeffs[0] * lambda + coeffs[1]) * lambda + coeffs[2];
        double dx[3] = { lambda * lambda, lambda, 1.0 };

        /* Sigmoid and its derivative (both share q = 1 / sqrt(1 + x^2)) */
        double q  = 1.0 / std::sqrt(1.0 + x * x);
        double s  = 0.5 * x * q + 0.5;
        double sp = 0.5 * q * q * q;

        /* Integrate the curves and their coefficient sensitivities */
        for (int j = 0; j < 3; ++j) {
            double w = rgb_tbl[j][i];
            out[j] += w * s;
            double wsp = w * sp;
            dout[j][0] += wsp * dx[0];
            dout[j][1] += wsp * dx[1];
            dout[j][2] += wsp * dx[2];
        }
    }

    /* Residual in CIELab */
    double out_lab[3] = { out[0], out[1], out[2] };
    cie_lab(out_lab);
    memcpy(residual, rgb, sizeof(double) * 3);
    cie_lab(residual);
    for (int j = 0; j < 3; ++j)
        residual[j] -= out_lab[j];

    /* Chain rule: d residual / d coeffs = -(d Lab / d out) * (d out / d coeffs) */
    double lab_jac[3][3];
    cie_lab_jac(out, lab_jac);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
                s += lab_jac[a][k] * dout[k][b];
            jac[a][b] = -s;
        }
}

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
double LM(const double rgb[3], double coeffs[3], int it = 15) {
    double residual[3], jac[3][3];
    eval_residual_jac(coeffs, rgb, residual, jac);
    double cost = sqr(residual[0]) + sqr(residual[1]) + sqr(residual[2]);

    double lambda = 1e-3;

    for (int i = 0; i < it && cost > 1e-12; ++i) {
        /* Assemble the normal equations: A = J^T J,  g = J^T residual */
        double A[3][3], g[3] = { 0.0, 0.0, 0.0 };
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double s = 0.0;
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
            double M0[3], M1[3], M2[3], *M[3] = { M0, M1, M2 };
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    M[a][b] = A[a][b] + (a == b ? lambda : 0.0); /* A + lambda*I */

            int P[4];
            if (LUPDecompose(M, 3, 1e-30, P) != 1) {
                lambda *= 10.0; /* singular even when damped -> damp harder */
                continue;
            }

            /* Solve (A + lambda*I) step = g;  the LM update is coeffs -= step.
               Trials only need the cost, so use the cheaper residual-only eval. */
            double step[3];
            LUPSolve(M, P, g, 3, step);

            double trial[3] = { coeffs[0] - step[0],
                                coeffs[1] - step[1],
                                coeffs[2] - step[2] };
            double trial_res[3];
            eval_residual(trial, rgb, trial_res);
            double trial_cost = sqr(trial_res[0]) + sqr(trial_res[1]) +
                                sqr(trial_res[2]);

            if (trial_cost < cost) {
                memcpy(coeffs, trial, sizeof(double) * 3);
                cost = trial_cost;
                lambda = std::max(lambda * 0.5, 1e-12); /* step worked: trust the model more */
                accepted = true;
            } else {
                lambda *= 10.0; /* step failed: trust the model less */
                if (lambda > 1e12)
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
        scale[k] = (float) smoothstep(smoothstep(k / double(res - 1)));

    size_t bufsize = 3*3*res*res*res;
    float *out = new float[bufsize];

    /* Each (l, j) slice is an independent unit of work: distinct slices write to
       disjoint regions of 'out', so the tasks need no synchronization. */
    auto process = [&](int l, int j) {
        const double y = j / double(res - 1);
        printf(".");
        fflush(stdout);
        for (int i = 0; i < res; ++i) {
            const double x = i / double(res - 1);
            double coeffs[3], rgb[3];
            memset(coeffs, 0, sizeof(double)*3);

            int start = res / 5;

            for (int k = start; k < res; ++k) {
                double b = (double) scale[k];

                rgb[l] = b;
                rgb[(l + 1) % 3] = x*b;
                rgb[(l + 2) % 3] = y*b;

                double resid = LM(rgb, coeffs);
                (void) resid;

                double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                double A = coeffs[0], B = coeffs[1], C = coeffs[2];

                int idx = ((l*res + k) * res + j)*res+i;

                out[3*idx + 0] = float(A*(sqr(c1)));
                out[3*idx + 1] = float(B*c1 - 2*A*c0*(sqr(c1)));
                out[3*idx + 2] = float(C - B*c0*c1 + A*(sqr(c0*c1)));
                //out[3*idx + 2] = resid;
            }

            memset(coeffs, 0, sizeof(double)*3);
            for (int k = start; k>=0; --k) {
                double b = (double) scale[k];

                rgb[l] = b;
                rgb[(l + 1) % 3] = x*b;
                rgb[(l + 2) % 3] = y*b;

                double resid = LM(rgb, coeffs);
                (void) resid;

                double c0 = 360.0, c1 = 1.0 / (830.0 - 360.0);
                double A = coeffs[0], B = coeffs[1], C = coeffs[2];

                int idx = ((l*res + k) * res + j)*res+i;

                out[3*idx + 0] = float(A*(sqr(c1)));
                out[3*idx + 1] = float(B*c1 - 2*A*c0*(sqr(c1)));
                out[3*idx + 2] = float(C - B*c0*c1 + A*(sqr(c0*c1)));
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
    delete[] out;
    delete[] scale;
    fclose(f);
    printf(" done.\n");
}

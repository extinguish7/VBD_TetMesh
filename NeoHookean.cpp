#include "NeoHookean.h"
#include <iostream>

namespace VBD {

    void NeoHookean::compute_Energy(
        Real& E,
        const MatrixXr& F,
        const Vec3& sigma,
        const Matrix3r& U,
        const Matrix3r& V,
        const std::vector<Real>& material_datas_vector
    ) {
        Real mu = material_datas_vector[0];
        Real lambda = material_datas_vector[1];
        const Real J = sigma.prod();
        E = mu * 0.5 * ((F.transpose() * F).trace() - 3)
            - mu * std::log(J)
            + 0.5 * lambda * std::pow(std::log(J), 2.0);
    }

    void NeoHookean::compute_dE_div_dF(
        Matrix3r& dE_div_dF,
        const MatrixXr& F,
        const Vec3& sigma,
        const Matrix3r& U,
        const Matrix3r& V,
        const std::vector<Real>& material_datas_vector
    ) {
        Real mu = material_datas_vector[0];
        Real lambda = material_datas_vector[1];
        const Real J = sigma.prod();

        Matrix3r FInvT;
        // 计算余子式矩阵 (F 的逆的转置 * J)
        FInvT(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
        FInvT(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
        FInvT(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
        FInvT(1, 0) = F(0, 2) * F(2, 1) - F(0, 1) * F(2, 2);
        FInvT(1, 1) = F(0, 0) * F(2, 2) - F(0, 2) * F(2, 0);
        FInvT(1, 2) = F(0, 1) * F(2, 0) - F(0, 0) * F(2, 1);
        FInvT(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
        FInvT(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
        FInvT(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);

        FInvT /= J;

        dE_div_dF = mu * (F - FInvT) + lambda * std::log(J) * FInvT;
    }

    void NeoHookean::compute_BLeftCoef(
        Vec3& BLeftCoef,
        const Vec3& sigma,
        const std::vector<Real>& material_datas_vector
    ) {
        Real mu = material_datas_vector[0];
        Real lambda = material_datas_vector[1];

        if (mu == 0.0 && lambda == 0.0) {
            BLeftCoef.setZero();
            return;
        }

        const Real sigmaProd = sigma.prod();
        const Real middle = mu - lambda * std::log(sigmaProd);

        BLeftCoef[0] = (mu + middle / sigma[0] / sigma[1]) / 2.0;
        BLeftCoef[1] = (mu + middle / sigma[1] / sigma[2]) / 2.0;
        BLeftCoef[2] = (mu + middle / sigma[2] / sigma[0]) / 2.0;
    }

    void NeoHookean::compute_dE_div_dsigma(
        Vec3& dE_div_dsigma,
        const Vec3& sigma,
        const std::vector<Real>& material_datas_vector
    ) {
        Real mu = material_datas_vector[0];
        Real lambda = material_datas_vector[1];
        const Real log_sigmaProd = std::log(sigma.prod());

        dE_div_dsigma.setZero();

        const Real inv0 = 1.0 / sigma[0];
        dE_div_dsigma[0] = mu * (sigma[0] - inv0) + lambda * inv0 * log_sigmaProd;

        const Real inv1 = 1.0 / sigma[1];
        dE_div_dsigma[1] = mu * (sigma[1] - inv1) + lambda * inv1 * log_sigmaProd;

        const Real inv2 = 1.0 / sigma[2];
        dE_div_dsigma[2] = mu * (sigma[2] - inv2) + lambda * inv2 * log_sigmaProd;
    }

    void NeoHookean::compute_d2E_div_dsigma2(
        Matrix3r& d2E_div_dsigma2,
        const Vec3& sigma,
        const std::vector<Real>& material_datas_vector
    ) {
        Real mu = material_datas_vector[0];
        Real lambda = material_datas_vector[1];
        const Real log_sigmaProd = std::log(sigma.prod());

        d2E_div_dsigma2.setZero();

        const Real inv2_0 = 1.0 / sigma[0] / sigma[0];
        d2E_div_dsigma2(0, 0) = mu * (1.0 + inv2_0) - lambda * inv2_0 * (log_sigmaProd - 1.0);

        const Real inv2_1 = 1.0 / sigma[1] / sigma[1];
        d2E_div_dsigma2(1, 1) = mu * (1.0 + inv2_1) - lambda * inv2_1 * (log_sigmaProd - 1.0);
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda / sigma[0] / sigma[1];

        const Real inv2_2 = 1.0 / sigma[2] / sigma[2];
        d2E_div_dsigma2(2, 2) = mu * (1.0 + inv2_2) - lambda * inv2_2 * (log_sigmaProd - 1.0);
        d2E_div_dsigma2(1, 2) = d2E_div_dsigma2(2, 1) = lambda / sigma[1] / sigma[2];
        d2E_div_dsigma2(2, 0) = d2E_div_dsigma2(0, 2) = lambda / sigma[2] / sigma[0];
    }

    void NeoHookean::makePD(Matrix3r& m) {
        Eigen::SelfAdjointEigenSolver<Matrix3r> es(m);
        Vec3 eigenvalues = es.eigenvalues();
        for (int i = 0; i < 3; i++) {
            if (eigenvalues[i] < EPSILON) {
                eigenvalues[i] = EPSILON;
            }
        }
        m = es.eigenvectors() * eigenvalues.asDiagonal() * es.eigenvectors().transpose();
    }

    void NeoHookean::makePD2d(Mat2& m) {
        Eigen::SelfAdjointEigenSolver<Mat2> es(m);
        Vec2 eigenvalues = es.eigenvalues();
        for (int i = 0; i < 2; i++) {
            if (eigenvalues[i] < EPSILON) {
                eigenvalues[i] = EPSILON;
            }
        }
        m = es.eigenvectors() * eigenvalues.asDiagonal() * es.eigenvectors().transpose();
    }

    void NeoHookean::compute_SVD(
        Matrix3r& U,
        Vec3& Sigma,
        Matrix3r& V,
        const Matrix3r& F
    ) {
        Eigen::JacobiSVD<Matrix3r> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        V = svd.matrixV();
        Sigma = svd.singularValues();

        // 确保奇异值非负
        for (int i = 0; i < 3; i++) {
            if (Sigma[i] < 0.0) Sigma[i] = 0.0;
        }
    }

    void NeoHookean::compute_dP_div_dF(
        Matrix9r& dP_div_dF,
        const MatrixXr& F,
        const Vec3& sigma,
        const Matrix3r& U,
        const Matrix3r& V,
        const std::vector<Real>& material_datas_vector
    ) {
        Real mu = material_datas_vector[0];
        Real lambda = material_datas_vector[1];

        //g_perfTimer.start("compute_dE_div_dsigma");
        // 计算 A 矩阵 (d2E_div_dsigma2)
        Vec3 dE_div_dsigma;
        compute_dE_div_dsigma(dE_div_dsigma, sigma, material_datas_vector);
        //g_perfTimer.end();

        //g_perfTimer.start("compute_d2E_div_dsigma2");
        Matrix3r d2E_div_dsigma2;
        compute_d2E_div_dsigma2(d2E_div_dsigma2, sigma, material_datas_vector);
        //g_perfTimer.end();

        //g_perfTimer.start("makePD d2E_div_dsigma2");
        makePD(d2E_div_dsigma2);
        //g_perfTimer.end();

        //g_perfTimer.start("compute_BLeftCoef");
        // 计算 B12, B13, B23
        const int Cdim2 = 3;
        Vec3 BLeftCoef;
        compute_BLeftCoef(BLeftCoef, sigma, material_datas_vector);

        Mat2 B[Cdim2];
        for (int cI = 0; cI < Cdim2; cI++) {
            int cI_post = (cI + 1) % 3;
            Real rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
            Real sum_sigma = sigma[cI] + sigma[cI_post];
            const Real eps = 1.0e-6;

            if (sum_sigma < eps) {
                rightCoef /= 2.0 * eps;
            }
            else {
                rightCoef /= 2.0 * sum_sigma;
            }

            const Real& leftCoef = BLeftCoef[cI];
            B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
            B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
            makePD2d(B[cI]);
        }
        //g_perfTimer.end();

        //g_perfTimer.start("compute_M");
        // 构建 M 矩阵 (9x9)
        Matrix9r M;
        M.setZero();

        // A 部分 (对角块)
        M(0, 0) = d2E_div_dsigma2(0, 0);
        M(0, 4) = d2E_div_dsigma2(0, 1);
        M(0, 8) = d2E_div_dsigma2(0, 2);
        M(4, 0) = d2E_div_dsigma2(1, 0);
        M(4, 4) = d2E_div_dsigma2(1, 1);
        M(4, 8) = d2E_div_dsigma2(1, 2);
        M(8, 0) = d2E_div_dsigma2(2, 0);
        M(8, 4) = d2E_div_dsigma2(2, 1);
        M(8, 8) = d2E_div_dsigma2(2, 2);

        // B01
        M(1, 1) = B[0](0, 0);
        M(1, 3) = B[0](0, 1);
        M(3, 1) = B[0](1, 0);
        M(3, 3) = B[0](1, 1);

        // B12
        M(5, 5) = B[1](0, 0);
        M(5, 7) = B[1](0, 1);
        M(7, 5) = B[1](1, 0);
        M(7, 7) = B[1](1, 1);

        // B20
        M(2, 2) = B[2](1, 1);
        M(2, 6) = B[2](1, 0);
        M(6, 2) = B[2](0, 1);
        M(6, 6) = B[2](0, 0);

        // 计算 dP_div_dF
        for (int i = 0; i < 3; i++) {
            int _dim_i = i * 3;
            for (int j = 0; j < 3; j++) {
                int ij = _dim_i + j;
                for (int r = 0; r < 3; r++) {
                    int _dim_r = r * 3;
                    for (int s = 0; s < 3; s++) {
                        int rs = _dim_r + s;
                        if (ij > rs) continue;

                        dP_div_dF(ij, rs) =
                            M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                            M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                            M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                            M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                            M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                            M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                            M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                            M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                            M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                            M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                            M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                            M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                            M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                            M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                            M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                            M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                            M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                            M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                            M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                            M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                            M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);

                        if (ij < rs) {
                            dP_div_dF(rs, ij) = dP_div_dF(ij, rs);
                        }
                    }
                }
            }
        }
        //g_perfTimer.end();

    }

} // namespace VBD
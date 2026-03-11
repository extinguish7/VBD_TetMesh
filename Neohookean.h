#pragma once

#include "Types.h"
#include <vector>

namespace VBD {

    /**
     * @brief Neo-Hookean 能量模型
     *
     * 实现论文中的 Neo-Hookean 材料能量、梯度和海森矩阵计算
     * 能量公式: W = μ/2 * (tr(F^T F) - 3) - μ * ln(J) + λ/2 * (ln(J))^2
     */
    class NeoHookean {
    public:
        /**
         * @brief 计算 Neo-Hookean 能量
         *
         * @param E 输出能量值
         * @param F 变形梯度 (3x3)
         * @param sigma 奇异值 (3x1)
         * @param U SVD 的 U 矩阵
         * @param V SVD 的 V 矩阵
         * @param material_datas_vector 材料参数 [mu, lambda]
         */
        static void compute_Energy(
            Real& E,
            const MatrixXr& F,
            const Vec3& sigma,
            const Matrix3r& U,
            const Matrix3r& V,
            const std::vector<Real>& material_datas_vector
        );

        /**
         * @brief 计算能量对 F 的一阶导数 (第一 Piola-Kirchhoff 应力 P)
         *
         * @param dE_div_dF 输出 ∂E/∂F (3x3)
         * @param F 变形梯度
         * @param sigma 奇异值
         * @param U SVD 的 U 矩阵
         * @param V SVD 的 V 矩阵
         * @param material_datas_vector 材料参数 [mu, lambda]
         */
        static void compute_dE_div_dF(
            Matrix3r& dE_div_dF,
            const MatrixXr& F,
            const Vec3& sigma,
            const Matrix3r& U,
            const Matrix3r& V,
            const std::vector<Real>& material_datas_vector
        );

        /**
         * @brief 计算应力对 F 的二阶导数 (9x9 矩阵)
         *
         * @param dP_div_dF 输出 ∂P/∂F (9x9)
         * @param F 变形梯度
         * @param sigma 奇异值
         * @param U SVD 的 U 矩阵
         * @param V SVD 的 V 矩阵
         * @param material_datas_vector 材料参数 [mu, lambda]
         */
        static void compute_dP_div_dF(
            Matrix9r& dP_div_dF,
            const MatrixXr& F,
            const Vec3& sigma,
            const Matrix3r& U,
            const Matrix3r& V,
            const std::vector<Real>& material_datas_vector
        );

        /**
         * @brief 计算能量对奇异值的一阶导数
         */
        static void compute_dE_div_dsigma(
            Vec3& dE_div_dsigma,
            const Vec3& sigma,
            const std::vector<Real>& material_datas_vector
        );

        /**
         * @brief 计算能量对奇异值的二阶导数
         */
        static void compute_d2E_div_dsigma2(
            Matrix3r& d2E_div_dsigma2,
            const Vec3& sigma,
            const std::vector<Real>& material_datas_vector
        );

        /**
         * @brief 计算 SVD 分解
         */
        static void compute_SVD(
            Matrix3r& U,
            Vec3& Sigma,
            Matrix3r& V,
            const Matrix3r& F
        );

        /**
         * @brief 确保矩阵正定
         */
        static void makePD(Matrix3r& m);
        static void makePD2d(Mat2& m);

    private:
        /**
         * @brief 计算 B 矩阵的左系数
         */
        static void compute_BLeftCoef(
            Vec3& BLeftCoef,
            const Vec3& sigma,
            const std::vector<Real>& material_datas_vector
        );
    };

} // namespace VBD
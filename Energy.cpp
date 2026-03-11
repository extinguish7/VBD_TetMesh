#include "Energy.h"
#include <omp.h>

namespace VBD {

    void ElasticEnergy::compute_Dm(
        Matrix3r& Dm,
        const TVerticesMat& x,
        const TetVIdsArr& tet
    ) {
        Dm.col(0) = x.col(tet[1]) - x.col(tet[0]);
        Dm.col(1) = x.col(tet[2]) - x.col(tet[0]);
        Dm.col(2) = x.col(tet[3]) - x.col(tet[0]);
    }

    void ElasticEnergy::compute_F(
        Matrix3r& F,
        const Matrix3r& Ds,
        const Matrix3r& DmInv
    ) {
        F = Ds * DmInv;
    }

    void ElasticEnergy::compute_Volume(
        Real& volume,
        const Matrix3r& Ds
    ) {
        Vec3 diff1 = Ds.col(0);
        Vec3 diff2 = Ds.col(1);
        Vec3 diff3 = Ds.col(2);
        volume = std::abs(diff3.dot(diff1.cross(diff2))) / 6.0f;
    }

    // 辅助函数：安全计算 log(J)，防止 J <= 0 导致 NaN
    inline Real safe_log(Real J, Real& d_log_J, Real& d2_log_J) {
        // 稳定性阈值，当 J 过小时钳位，保证数值稳定
        const Real J_min = 0.1;
        Real J_clamped = J;
        bool clamped = false;

        if (J < J_min) {
            J_clamped = J_min;
            clamped = true;
        }

        Real log_J = std::log(J_clamped);

        // 导数
        d_log_J = 1.0 / J_clamped;
        // 二阶导数
        d2_log_J = -1.0 / (J_clamped * J_clamped);

        // 如果发生了钳位，为了 C2 连续性通常需要做平滑过渡，
        // 但在实时仿真中，直接截断导数通常足够"稳定"且防止崩溃。
        // 这里为了简单和鲁棒性，直接使用钳位后的导数。

        return log_J;
    }

    // 辅助函数：计算 3x3 矩阵的代数余子式矩阵 (Cofactor Matrix)
    static Matrix3r computeCofactor(const Matrix3r& F) {
        Matrix3r C;
        C(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
        C(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
        C(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
        C(1, 0) = F(2, 1) * F(0, 2) - F(2, 2) * F(0, 1);
        C(1, 1) = F(2, 2) * F(0, 0) - F(2, 0) * F(0, 2);
        C(1, 2) = F(2, 0) * F(0, 1) - F(2, 1) * F(0, 0);
        C(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
        C(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
        C(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
        return C;
    }

    // 辅助函数：计算代数余子式矩阵的方向导数 dC(F, dF)
    static Matrix3r computeCofactorDerivative(const Matrix3r& F, const Matrix3r& dF) {
        Matrix3r dC;
        dC(0, 0) = dF(1, 1) * F(2, 2) + F(1, 1) * dF(2, 2) - dF(1, 2) * F(2, 1) - F(1, 2) * dF(2, 1);
        dC(0, 1) = dF(1, 2) * F(2, 0) + F(1, 2) * dF(2, 0) - dF(1, 0) * F(2, 2) - F(1, 0) * dF(2, 2);
        dC(0, 2) = dF(1, 0) * F(2, 1) + F(1, 0) * dF(2, 1) - dF(1, 1) * F(2, 0) - F(1, 1) * dF(2, 0);
        dC(1, 0) = dF(2, 1) * F(0, 2) + F(2, 1) * dF(0, 2) - dF(2, 2) * F(0, 1) - F(2, 2) * dF(0, 1);
        dC(1, 1) = dF(2, 2) * F(0, 0) + F(2, 2) * dF(0, 0) - dF(2, 0) * F(0, 2) - F(2, 0) * dF(0, 2);
        dC(1, 2) = dF(2, 0) * F(0, 1) + F(2, 0) * dF(0, 1) - dF(2, 1) * F(0, 0) - F(2, 1) * dF(0, 0);
        dC(2, 0) = dF(0, 1) * F(1, 2) + F(0, 1) * dF(1, 2) - dF(0, 2) * F(1, 1) - F(0, 2) * dF(1, 1);
        dC(2, 1) = dF(0, 2) * F(1, 0) + F(0, 2) * dF(1, 0) - dF(0, 0) * F(1, 2) - F(0, 0) * dF(1, 2);
        dC(2, 2) = dF(0, 0) * F(1, 1) + F(0, 0) * dF(1, 1) - dF(0, 1) * F(1, 0) - F(0, 1) * dF(1, 0);
        return dC;
    }

    void ElasticEnergy::compute_Hessian_and_Gradient(Matrix12r& hessian, Vec12& gradient, const Matrix3r& Ds, const Matrix3r& DmInv, const Matrix3r& F, const Real& lambda, const Real& mu, const Real volume)
    {
        // 1. 预计算核心变量
        Real J = F.determinant();
        Matrix3r C = computeCofactor(F);

        // 2. 计算 First Piola-Kirchhoff 应力 P = \partial \Psi / \partial F (3x3 矩阵)
        Matrix3r P = mu * (F - C) + lambda * (J - 1.0) * C;

        // 3. 构建节点坐标向 F 的映射矩阵 Q (4x3 矩阵)
        // 根据定义 F = Ds * DmInv = X * Ps * DmInv
        Eigen::Matrix<Real, 4, 3> Ps;
        Ps << -1, -1, -1,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1;
        Eigen::Matrix<Real, 4, 3> Q = Ps * DmInv;

        // 4. 计算并组装节点梯度 (节点布局假定为：x0,y0,z0, x1,y1,z1 ...)
        Eigen::Matrix<Real, 3, 4> g_nodes = volume * P * Q.transpose();
        for (int i = 0; i < 4; ++i) {
            gradient.segment<3>(3 * i) = g_nodes.col(i);
        }

        // 5. 计算能量密度对 F 的 9x9 局部海森矩阵 (dF -> dP 的映射)
        Eigen::Matrix<Real, 9, 9> HF = Eigen::Matrix<Real, 9, 9>::Zero();
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                int col = i + 3 * j;
                Matrix3r dF = Matrix3r::Zero();
                dF(i, j) = 1.0;

                // Frobenius 内积: C : dF 等价于 dJ
                Real dJ = (C.array() * dF.array()).sum();
                Matrix3r dC = computeCofactorDerivative(F, dF);

                // 计算方向微分 dP
                Matrix3r dP = mu * (dF - dC) + lambda * dJ * C + lambda * (J - 1.0) * dC;

                // 将 3x3 的 dP 展平并存入 HF 的对应列
                for (int vj = 0; vj < 3; ++vj) {
                    for (int vi = 0; vi < 3; ++vi) {
                        HF(vi + 3 * vj, col) = dP(vi, vj);
                    }
                }
            }
        }

        // 6. 将 9x9 的材料海森矩阵通过链式法则映射为 12x12 的节点全局海森矩阵
        hessian.setZero();
        for (int n = 0; n < 4; ++n) {            // 遍历行节点 n
            for (int m = 0; m < 4; ++m) {        // 遍历列节点 m
                for (int c = 0; c < 3; ++c) {    // 节点 n 的 xyz 坐标
                    for (int c_prime = 0; c_prime < 3; ++c_prime) { // 节点 m 的 xyz 坐标
                        Real val = 0.0;
                        for (int b = 0; b < 3; ++b) {
                            for (int d = 0; d < 3; ++d) {
                                val += HF(c + 3 * b, c_prime + 3 * d) * Q(n, b) * Q(m, d);
                            }
                        }
                        hessian(3 * n + c, 3 * m + c_prime) = volume * val;
                    }
                }
            }
        }

		//std::cout << "gradient: " << gradient.transpose() << std::endl;
		//std::cout << "hessian:\n" << hessian << std::endl;
    }

} // namespace VBD
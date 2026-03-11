#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "Types.h"

namespace VBD_GPU {

    // 基础 3x3 矩阵与 3 维向量
    struct Mat3d { Real m[3][3]; };

    __device__ __forceinline__ Real det3x3(const Mat3d& A) {
        return A.m[0][0] * (A.m[1][1] * A.m[2][2] - A.m[1][2] * A.m[2][1])
            - A.m[0][1] * (A.m[1][0] * A.m[2][2] - A.m[1][2] * A.m[2][0])
            + A.m[0][2] * (A.m[1][0] * A.m[2][1] - A.m[1][1] * A.m[2][0]);
    }

    // 计算 3x3 的伴随矩阵
    __device__ __forceinline__ Mat3d computeCofactor(const Mat3d& F) {
        Mat3d C;
        C.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
        C.m[0][1] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
        C.m[0][2] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];
        C.m[1][0] = F.m[2][1] * F.m[0][2] - F.m[2][2] * F.m[0][1];
        C.m[1][1] = F.m[2][2] * F.m[0][0] - F.m[2][0] * F.m[0][2];
        C.m[1][2] = F.m[2][0] * F.m[0][1] - F.m[2][1] * F.m[0][0];
        C.m[2][0] = F.m[0][1] * F.m[1][2] - F.m[0][2] * F.m[1][1];
        C.m[2][1] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
        C.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];
        return C;
    }

    // 补齐：计算代数余子式的方向导数 (从CPU版的 Energy.cpp 移植)
    __device__ __forceinline__ Mat3d computeCofactorDerivative(const Mat3d& F, const Mat3d& dF) {
        Mat3d dC;
        dC.m[0][0] = dF.m[1][1] * F.m[2][2] + F.m[1][1] * dF.m[2][2] - dF.m[1][2] * F.m[2][1] - F.m[1][2] * dF.m[2][1];
        dC.m[0][1] = dF.m[1][2] * F.m[2][0] + F.m[1][2] * dF.m[2][0] - dF.m[1][0] * F.m[2][2] - F.m[1][0] * dF.m[2][2];
        dC.m[0][2] = dF.m[1][0] * F.m[2][1] + F.m[1][0] * dF.m[2][1] - dF.m[1][1] * F.m[2][0] - F.m[1][1] * dF.m[2][0];
        dC.m[1][0] = dF.m[2][1] * F.m[0][2] + F.m[2][1] * dF.m[0][2] - dF.m[2][2] * F.m[0][1] - F.m[2][2] * dF.m[0][1];
        dC.m[1][1] = dF.m[2][2] * F.m[0][0] + F.m[2][2] * dF.m[0][0] - dF.m[2][0] * F.m[0][2] - F.m[2][0] * dF.m[0][2];
        dC.m[1][2] = dF.m[2][0] * F.m[0][1] + F.m[2][0] * dF.m[0][1] - dF.m[2][1] * F.m[0][0] - F.m[2][1] * dF.m[0][0];
        dC.m[2][0] = dF.m[0][1] * F.m[1][2] + F.m[0][1] * dF.m[1][2] - dF.m[0][2] * F.m[1][1] - F.m[0][2] * dF.m[1][1];
        dC.m[2][1] = dF.m[0][2] * F.m[1][0] + F.m[0][2] * dF.m[1][0] - dF.m[0][0] * F.m[1][2] - F.m[0][0] * dF.m[1][2];
        dC.m[2][2] = dF.m[0][0] * F.m[1][1] + F.m[0][0] * dF.m[1][1] - dF.m[0][1] * F.m[1][0] - F.m[0][1] * dF.m[1][0];
        return dC;
    }

    // 求解 3x3 线性系统 H * dx = f
    __device__ __forceinline__ bool solve3x3(const Real H[9], const Real f[3], Real dx[3]) {
        Mat3d A;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) A.m[i][j] = H[i * 3 + j];

        Real det = det3x3(A);
        if (fabs(det) < 1.0e-8f) {
            dx[0] = dx[1] = dx[2] = 0.0f;
            return false;
        }

        Mat3d adj = computeCofactor(A);
        // 转置伴随矩阵即为真正的伴随阵
        Real invDet = 1.0f / det;
        for (int i = 0; i < 3; ++i) {
            dx[i] = invDet * (adj.m[0][i] * f[0] + adj.m[1][i] * f[1] + adj.m[2][i] * f[2]);
        }
        return true;
    }

    // ====== 核心优化：只提取特定节点 n (0-3) 的 3x3 海森矩阵和 3x1 梯度 ======
    // 补齐：完美将 12x12 -> 3x3 截断，且保留所有数学等价性
	__device__ __forceinline__ void computeTetContributionLocal(
		const Mat3d& F, const Mat3d& DmInv, Real mu, Real lambda, Real volume, int localNodeIdx,
		Real out_H[9], Real out_f[3])
    {
        Real J = det3x3(F);
        Mat3d C = computeCofactor(F);

        Mat3d P;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                P.m[i][j] = mu * (F.m[i][j] - C.m[i][j]) + lambda * (J - 1.0f) * C.m[i][j];
            }
        }

        Real Qn[3] = { 0.0, 0.0, 0.0 };
        if (localNodeIdx == 0) {
            Qn[0] = -DmInv.m[0][0] - DmInv.m[1][0] - DmInv.m[2][0];
            Qn[1] = -DmInv.m[0][1] - DmInv.m[1][1] - DmInv.m[2][1];
            Qn[2] = -DmInv.m[0][2] - DmInv.m[1][2] - DmInv.m[2][2];
        }
        else {
            Qn[0] = DmInv.m[localNodeIdx - 1][0];
            Qn[1] = DmInv.m[localNodeIdx - 1][1];
            Qn[2] = DmInv.m[localNodeIdx - 1][2];
        }

        // 提取梯度 f (3x1)
        for (int i = 0; i < 3; ++i) {
            out_f[i] = volume * (P.m[i][0] * Qn[0] + P.m[i][1] * Qn[1] + P.m[i][2] * Qn[2]);
        }

        for (int i = 0; i < 9; ++i) out_H[i] = 0.0f;

        // 补齐：双重循环求解 3x3 块
        for (int c_prime = 0; c_prime < 3; ++c_prime) {
            for (int d = 0; d < 3; ++d) {
                Mat3d dF;
                for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) dF.m[a][b] = 0.0f;
                dF.m[c_prime][d] = 1.0f;

                Mat3d dC = computeCofactorDerivative(F, dF);
                Real dJ = C.m[c_prime][d]; // F 和 dJ 的点积，由于 dF 只有一个 1，因此直接提取 C 的元素

                for (int c = 0; c < 3; ++c) {
                    Real val = 0.0f;
                    for (int b = 0; b < 3; ++b) {
                        Real dP_cb = mu * (dF.m[c][b] - dC.m[c][b])
                            + lambda * dJ * C.m[c][b]
                            + lambda * (J - 1.0f) * dC.m[c][b];
                        val += dP_cb * Qn[b] * Qn[d];
                    }
                    out_H[c * 3 + c_prime] += volume * val; // 行:c, 列:c_prime
                }
            }
        }
    }

    // ====== 新增：专为金属硬物设计的共旋线性弹性模型 (Corotated Linear Elasticity) ======
    __device__ __forceinline__ void computeTetContributionLocalCorotated(
        const Mat3d& F, const Mat3d& DmInv, Real mu, Real lambda, Real volume, int localNodeIdx,
        Real out_H[9], Real out_f[3])
    {
        // 1. 极分解 (Polar Decomposition): 提取纯旋转矩阵 R (使用极速的高推算法 Higham's Method)
        Mat3d R = F;
        for (int iter = 0; iter < 5; ++iter) {
            Real det = det3x3(R);
            if (fabs(det) < 1e-8f) break; // 防止高度压扁时奇异

            Mat3d C = computeCofactor(R); // R^{-T} = Cofactor(R) / det(R)
            Real invDet = 1.0f / det;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R.m[i][j] = 0.5f * (R.m[i][j] + C.m[i][j] * invDet);
                }
            }
        }

        // 2. 计算第一类 Piola-Kirchhoff 应力 P
        // 公式: P = 2*mu*(F - R) + lambda * tr(R^T * F - I) * R
        Real tr = 0.0f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                tr += R.m[i][j] * F.m[i][j];
            }
        }
        tr -= 3.0f; // 减去 I 的迹

        Mat3d P;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                P.m[i][j] = 2.0f * mu * (F.m[i][j] - R.m[i][j]) + lambda * tr * R.m[i][j];
            }
        }

        // 3. 计算伴随向量 Qn (形状函数的空间偏导)
        Real Qn[3] = { 0.0f, 0.0f, 0.0f };
        if (localNodeIdx == 0) {
            Qn[0] = -DmInv.m[0][0] - DmInv.m[1][0] - DmInv.m[2][0];
            Qn[1] = -DmInv.m[0][1] - DmInv.m[1][1] - DmInv.m[2][1];
            Qn[2] = -DmInv.m[0][2] - DmInv.m[1][2] - DmInv.m[2][2];
        }
        else {
            Qn[0] = DmInv.m[localNodeIdx - 1][0];
            Qn[1] = DmInv.m[localNodeIdx - 1][1];
            Qn[2] = DmInv.m[localNodeIdx - 1][2];
        }

        // 4. 计算顶点的力梯度 f (即能量关于 x_n 的一阶导，由于求解器是 f = -dE/dx，这里抽出的是 dE/dx)
        for (int i = 0; i < 3; ++i) {
            out_f[i] = volume * (P.m[i][0] * Qn[0] + P.m[i][1] * Qn[1] + P.m[i][2] * Qn[2]);
        }

        // 5. 【终极绝招】使用近似的 SPD Hessian 矩阵 (基于 Projective Dynamics 假定)
        // 忽略 dR/dF 的非线性项，得出的 Hessian 永远绝对正定，免疫所有数值爆炸！
        // 公式: H_nn = Volume * [ 2*mu*(Qn·Qn)*I + lambda*(R*Qn)⊗(R*Qn) ]
        Real Qn_sq = Qn[0] * Qn[0] + Qn[1] * Qn[1] + Qn[2] * Qn[2];

        Real RQ[3];
        for (int i = 0; i < 3; ++i) {
            RQ[i] = R.m[i][0] * Qn[0] + R.m[i][1] * Qn[1] + R.m[i][2] * Qn[2];
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real H_ij = (i == j ? 2.0f * mu * Qn_sq : 0.0f) + lambda * RQ[i] * RQ[j];
                out_H[i * 3 + j] = volume * H_ij;
            }
        }
    }
}

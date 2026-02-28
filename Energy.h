#pragma once

#include "Types.h"
#include <vector>
#include <map>
#include <tuple>

namespace VBD {

    // 类型别名用于矩阵乘法
    using Matrix12r9r = Eigen::Matrix<Real, 12, 9>;
    using Matrix9r12r = Eigen::Matrix<Real, 9, 12>;

    /**
     * @brief 弹性能量基类
     */
    class ElasticEnergy {
    public:
        virtual ~ElasticEnergy() = default;

        /**
        * @brief 计算 Stable Neo-Hookean 材料的梯度 (力向量) 和海森矩阵 (刚度矩阵)
        *
        * @param hessian 输出：12x12 海森矩阵 (刚度矩阵 K)
        * @param gradient 输出：12x1 梯度向量 (内力向量，负号视优化器定义而定，此处为 dE/dx)
        * @param Ds 当前构型边矩阵 (3x3), 列向量为 (x1-x0, x2-x0, x3-x0)
        * @param DmInv 参考构型边矩阵的逆 (3x3), 用于计算形状函数梯度
        * @param F 变形梯度 (3x3), F = Ds * DmInv
        * @param lambda 拉梅第一参数 (体积模量相关)
        * @param mu 拉梅第二参数 (剪切模量)
        * @param volume 参考体积 V0
        */
        static void compute_Hessian_and_Gradient(
            Matrix12r& hessian,
            Vec12& gradient,
            const Matrix3r& Ds,
            const Matrix3r& DmInv,
            const Matrix3r& F,
            const Real& lambda,
            const Real& mu,
            const Real volume
        );

        /**
         * @brief 计算参考配置下的边向量矩阵 Dm
         */
        static void compute_Dm(
            Matrix3r& Dm,
            const TVerticesMat& x,
            const TetVIdsArr& tet
        );

        /**
         * @brief 计算变形梯度 F = Ds * DmInv
         */
        static void compute_F(
            Matrix3r& F,
            const Matrix3r& Ds,
            const Matrix3r& DmInv
        );

        /**
         * @brief 计算四面体体积
         */
        static void compute_Volume(
            Real& volume,
            const Matrix3r& Ds
        );

    };

} // namespace VBD
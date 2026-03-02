#pragma once

#include "Types.h"
#include "TetMesh.h"
#include "Energy.h"
#include <vector>
#include <functional>

namespace VBD {

    /**
     * @brief VBD 求解器参数
     */
    struct VBDParams {
        Real dt = 0.01666666f;      // 时间步长 (默认 60fps)
        int numIterations = 50;    // 每步迭代次数
        int substeps = 1;          // 子步数
        Vec3 gravity{ 0, -9.8f, 0 };  // 重力加速度
        bool useAcceleration = false;      // 是否使用 Chebyshev 加速
        Real accelerationRho = 0.75f;       // 加速谱半径估计
        int numThreads = 4;                // 并行线程数
        Real mu = 1e6f;             // Neo-Hookean 剪切模量
        Real lambda = 1e6f;         // Neo-Hookean 拉梅第一参数
		Real density = 1000.0f;       // 材料密度
    };

    /**
     * @brief Vertex Block Descent 求解器
     *
     * 实现论文 "Vertex Block Descent" 的核心算法
     * 用于求解 Neo-Hookean 弹性体的隐式欧拉积分
     */
    class VBDsolver {
    public:
        VBDsolver();
        ~VBDsolver();

        /**
         * @brief 设置求解器参数
         */
        void setParams(const VBDParams& params);


        /**
         * @brief 初始化顶点着色
         */
        void initializeColoring(const TetMesh& mesh);

        /**
         * @brief 预计算四面体数据 (DmInv, 静止体积)
         */
        void precomputeTetData(const TetMesh& mesh);


        /**
         * @brief 执行一步模拟
         */
        void step(TetMesh& mesh);

        /**
         * @brief 执行多帧模拟
         */
        void simulate(TetMesh& mesh, int numFrames,
            std::function<void(int, TetMesh&)> callback = nullptr);

        /**
         * @brief 获取当前总能量
         */
        Real computeTotalEnergy(const TetMesh& mesh) const;

        /**
         * @brief 获取颜色分组信息
         */
        const std::vector<std::vector<IdType>>& getVertexColors() const {
            return vertexColors;
        }

		const std::vector<Matrix3r>& getTetDmInv() const {
			return tetDmInv;
		}

        const std::vector<Real>& getTetRestVol() const {
			return tetRestVolume;
        }

    public:
        VBDParams params;
        std::vector<std::vector<IdType>> vertexColors;  // 顶点颜色分组
        TVerticesMat prevPrevPos;                       // 用于加速的前两帧位置
        std::vector<Real> energyHistory;                // 能量历史

        // 每个四面体的预计算数据
        std::vector<Matrix3r> tetDmInv;     // Dm 的逆
        std::vector<Real> tetRestVolume;    // 静止体积

        /**
         * @brief 前向步（计算惯性项）
         */
        void forwardStep(TetMesh& mesh);

        /**
         * @brief 求解局部系统（核心 VBD 迭代）
         */
        void solve(TetMesh& mesh);

        /**
         * @brief 对单个顶点求解局部系统
         *
         * 计算该顶点的 3x3 海森矩阵 H 和 3x1 力向量 f
         */
        void solveVertex(
            const TetMesh& mesh,
            IdType vertexId,
            Mat3& H,
            Vec3& f,
            Real dtSqrReciprocal
        );

        /**
         * @brief 计算四面体对顶点的梯度和海森贡献
         */
        void computeTetContribution(
            const TetMesh& mesh,
            IdType tetId,
            IdType vertexId,
            int localVertexIdx,
            Mat3& H_local,
            Vec3& f_local
        );

        /**
         * @brief Chebyshev 加速
         */
        void applyAcceleration(TetMesh& mesh, Real omega);

        /**
         * @brief 获取加速系数
         */
        Real getAcceleratorOmega(int order, Real rho, Real prevOmega);
    };

} // namespace VBD
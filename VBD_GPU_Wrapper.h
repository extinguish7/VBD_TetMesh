#pragma once
#include "TetMesh.h"
#include <vector>

namespace VBD {

    class VBD_GPU_Solver {
    public:
        VBD_GPU_Solver() : d_positions(nullptr), d_inertia(nullptr), d_masses(nullptr),
            d_tetDmInv(nullptr), d_tetRestVol(nullptr), d_tets(nullptr),
            d_is_fixed(nullptr), d_adj_offsets(nullptr), d_adj_data(nullptr) {}
        ~VBD_GPU_Solver();

        // 从 CPU 初始化显存
        void initFromCPU(const TetMesh& mesh, const std::vector<std::vector<int>>& vertexColors,
            const std::vector<Matrix3r>& tetDmInv, const std::vector<Real>& tetRestVol);

        // 运行迭代并更新位置
        void solveIteration(Real dtSqrReciprocal, Real mu, Real lambda);

        // 将 GPU 位置拷回 CPU
        void copyPositionsToCPU(TetMesh& mesh);

        // 【新增】将最新的惯性位置同步到 GPU
        void updateInertiaFromCPU(const TetMesh& mesh);

        // 【新增】包含 Chebyshev 加速的完整迭代求解循环
        void solveVBD(int numIterations, Real dtSqrReciprocal, Real mu, Real lambda,
            bool useAcceleration = true, Real rho = 0.99f);

    private:
        Real* d_positions, * d_inertia, * d_masses, * d_tetDmInv, * d_tetRestVol;
        int* d_tets, * d_adj_offsets, * d_adj_data;
        bool* d_is_fixed;
        int numVerts;

        // 【新增】用于 Chebyshev 加速的历史位置缓冲区
        Real* d_prevPrevPos, * d_currentPos;

        struct ColorGroup {
            int* d_vertices;
            int num_vertices;
        };
        std::vector<ColorGroup> colorGroups;
    };

}
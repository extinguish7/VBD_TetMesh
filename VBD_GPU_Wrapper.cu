#include "VBD_DeviceMath.cuh"
#include "VBD_GPU_Wrapper.h"
#include <iostream>

using namespace VBD_GPU;

// 【新增】步骤A：Chebyshev 加速的 CUDA 内核
__global__ void vbd_apply_chebyshev_kernel(
    Real* d_positions, const Real* d_prevPrevPos,
    int num_verts, Real omega, const bool* d_is_fixed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;
    if (d_is_fixed[i]) return; // 固定点不加速

    // Eq: x_new = omega * (x_new - x_prev_prev) + x_prev_prev
    d_positions[i * 3 + 0] = omega * (d_positions[i * 3 + 0] - d_prevPrevPos[i * 3 + 0]) + d_prevPrevPos[i * 3 + 0];
    d_positions[i * 3 + 1] = omega * (d_positions[i * 3 + 1] - d_prevPrevPos[i * 3 + 1]) + d_prevPrevPos[i * 3 + 1];
    d_positions[i * 3 + 2] = omega * (d_positions[i * 3 + 2] - d_prevPrevPos[i * 3 + 2]) + d_prevPrevPos[i * 3 + 2];
}

//__global__ void vbd_solve_color_kernel(
//    const int* d_color_vertices, int num_vertices_in_color, Real* d_positions,
//    const Real* d_inertia, const Real* d_masses, const int* d_adj_offsets,
//    const int* d_adj_data, const int* d_tets, const Real* d_tetDmInv,
//    const Real* d_tetRestVol, const bool* d_is_fixed,
//    Real mu, Real lambda, Real dtSqrReciprocal)
//{
//    int color_idx = blockIdx.x;
//    if (color_idx >= num_vertices_in_color) return;
//
//    int v_id = d_color_vertices[color_idx];
//    if (d_is_fixed[v_id]) return;
//
//    __shared__ Real s_H[9];
//    __shared__ Real s_f[3];
//
//    // 初始化对角海森和惯性力
//    if (threadIdx.x == 0) {
//        Real m = d_masses[v_id];
//        Real coeff = m * dtSqrReciprocal;
//
//        for (int i = 0; i < 9; ++i) s_H[i] = 0.0f;
//        s_H[0] = coeff; s_H[4] = coeff; s_H[8] = coeff;
//
//        s_f[0] = coeff * (d_inertia[v_id * 3 + 0] - d_positions[v_id * 3 + 0]);
//        s_f[1] = coeff * (d_inertia[v_id * 3 + 1] - d_positions[v_id * 3 + 1]);
//        s_f[2] = coeff * (d_inertia[v_id * 3 + 2] - d_positions[v_id * 3 + 2]);
//    }
//    __syncthreads();
//
//    int start_adj = d_adj_offsets[v_id];
//    int end_adj = d_adj_offsets[v_id + 1];
//    int num_adj_tets = end_adj - start_adj;
//
//    // 线程块内并发遍历该顶点周围所有的四面体
//    for (int i = threadIdx.x; i < num_adj_tets; i += blockDim.x) {
//        int tet_id = d_adj_data[start_adj + i];
//
//        int localIdx = -1;
//        int v0 = d_tets[tet_id * 4 + 0], v1 = d_tets[tet_id * 4 + 1], v2 = d_tets[tet_id * 4 + 2], v3 = d_tets[tet_id * 4 + 3];
//        if (v0 == v_id) localIdx = 0;
//        else if (v1 == v_id) localIdx = 1;
//        else if (v2 == v_id) localIdx = 2;
//        else if (v3 == v_id) localIdx = 3;
//
//        if (localIdx != -1) {
//            // 获取四面体节点当前位置
//            Real p0[3] = { d_positions[v0 * 3], d_positions[v0 * 3 + 1], d_positions[v0 * 3 + 2] };
//            Real p1[3] = { d_positions[v1 * 3], d_positions[v1 * 3 + 1], d_positions[v1 * 3 + 2] };
//            Real p2[3] = { d_positions[v2 * 3], d_positions[v2 * 3 + 1], d_positions[v2 * 3 + 2] };
//            Real p3[3] = { d_positions[v3 * 3], d_positions[v3 * 3 + 1], d_positions[v3 * 3 + 2] };
//
//            // 计算变形矩阵 Ds
//            Mat3d Ds;
//            Ds.m[0][0] = p1[0] - p0[0]; Ds.m[0][1] = p2[0] - p0[0]; Ds.m[0][2] = p3[0] - p0[0];
//            Ds.m[1][0] = p1[1] - p0[1]; Ds.m[1][1] = p2[1] - p0[1]; Ds.m[1][2] = p3[1] - p0[1];
//            Ds.m[2][0] = p1[2] - p0[2]; Ds.m[2][1] = p2[2] - p0[2]; Ds.m[2][2] = p3[2] - p0[2];
//
//            Mat3d DmInv;
//            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) DmInv.m[a][b] = d_tetDmInv[tet_id * 9 + a * 3 + b];
//
//            // 计算 F = Ds * DmInv
//            Mat3d F;
//            for (int a = 0; a < 3; ++a) {
//                for (int b = 0; b < 3; ++b) {
//                    F.m[a][b] = Ds.m[a][0] * DmInv.m[0][b] + Ds.m[a][1] * DmInv.m[1][b] + Ds.m[a][2] * DmInv.m[2][b];
//                }
//            }
//
//            Real local_H[9], local_f[3];
//            Real vol = d_tetRestVol[tet_id];
//
//            computeTetContributionLocal(F, DmInv, mu, lambda, vol, localIdx, local_H, local_f);
//
//            // 通过共享内存上的原子加法，避免线程数据写冲突
//            for (int k = 0; k < 9; ++k) atomicAdd(&s_H[k], local_H[k]);
//            for (int k = 0; k < 3; ++k) atomicAdd(&s_f[k], -local_f[k]);
//        }
//    }
//    __syncthreads();
//
//    // 线程0最后汇总求逆，更新中心顶点坐标
//    if (threadIdx.x == 0) {
//        Real dx[3];
//        if (solve3x3(s_H, s_f, dx)) {
//            d_positions[v_id * 3 + 0] += dx[0];
//            d_positions[v_id * 3 + 1] += dx[1];
//            d_positions[v_id * 3 + 2] += dx[2];
//        }
//    }
//}

__global__ void vbd_solve_color_kernel(
    const int* d_color_vertices, int num_vertices_in_color, Real* d_positions,
    const Real* d_inertia, const Real* d_masses, const int* d_adj_offsets,
    const int* d_adj_data, const int* d_tets, const Real* d_tetDmInv,
    const Real* d_tetRestVol, const bool* d_is_fixed,
    Real mu, Real lambda, Real dtSqrReciprocal)
{
    int color_idx = blockIdx.x;
    if (color_idx >= num_vertices_in_color) return;

    int v_id = d_color_vertices[color_idx];
    if (d_is_fixed[v_id]) return;

    // 分配二维共享内存，每个线程对应自己的一组 9 和 3。
    // 这里假设你在主机端 launch kernel 时设置的 threads_per_block 不超过 32 (即一个 Warp)。
    // 如果你用的线程数是 16，这里分配 32 也能向下兼容。如果大于 32，请修改此处大小。
    __shared__ Real s_H_all[32][9];
    __shared__ Real s_f_all[32][3];

    // 线程私有寄存器，用于累加当前线程负责的多个四面体力
    // 在寄存器中累加的速度是极快的
    Real my_H[9] = { 0.0f };
    Real my_f[3] = { 0.0f };

    int start_adj = d_adj_offsets[v_id];
    int end_adj = d_adj_offsets[v_id + 1];
    int num_adj_tets = end_adj - start_adj;

    // 线程块内并发遍历该顶点周围所有的四面体
    for (int i = threadIdx.x; i < num_adj_tets; i += blockDim.x) {
        int tet_id = d_adj_data[start_adj + i];

        int localIdx = -1;
        int v0 = d_tets[tet_id * 4 + 0], v1 = d_tets[tet_id * 4 + 1], v2 = d_tets[tet_id * 4 + 2], v3 = d_tets[tet_id * 4 + 3];
        if (v0 == v_id) localIdx = 0;
        else if (v1 == v_id) localIdx = 1;
        else if (v2 == v_id) localIdx = 2;
        else if (v3 == v_id) localIdx = 3;

        if (localIdx != -1) {
            // 获取四面体节点当前位置
            Real p0[3] = { d_positions[v0 * 3], d_positions[v0 * 3 + 1], d_positions[v0 * 3 + 2] };
            Real p1[3] = { d_positions[v1 * 3], d_positions[v1 * 3 + 1], d_positions[v1 * 3 + 2] };
            Real p2[3] = { d_positions[v2 * 3], d_positions[v2 * 3 + 1], d_positions[v2 * 3 + 2] };
            Real p3[3] = { d_positions[v3 * 3], d_positions[v3 * 3 + 1], d_positions[v3 * 3 + 2] };

            // 计算变形矩阵 Ds
            Mat3d Ds;
            Ds.m[0][0] = p1[0] - p0[0]; Ds.m[0][1] = p2[0] - p0[0]; Ds.m[0][2] = p3[0] - p0[0];
            Ds.m[1][0] = p1[1] - p0[1]; Ds.m[1][1] = p2[1] - p0[1]; Ds.m[1][2] = p3[1] - p0[1];
            Ds.m[2][0] = p1[2] - p0[2]; Ds.m[2][1] = p2[2] - p0[2]; Ds.m[2][2] = p3[2] - p0[2];

            Mat3d DmInv;
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) DmInv.m[a][b] = d_tetDmInv[tet_id * 9 + a * 3 + b];

            // 计算 F = Ds * DmInv
            Mat3d F;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    F.m[a][b] = Ds.m[a][0] * DmInv.m[0][b] + Ds.m[a][1] * DmInv.m[1][b] + Ds.m[a][2] * DmInv.m[2][b];
                }
            }

            Real local_H[9], local_f[3];
            Real vol = d_tetRestVol[tet_id];

            computeTetContributionLocal(F, DmInv, mu, lambda, vol, localIdx, local_H, local_f);

            // 【优化核心】将计算结果累加到私有寄存器中，彻底移除原子操作
            for (int k = 0; k < 9; ++k) my_H[k] += local_H[k];
            for (int k = 0; k < 3; ++k) my_f[k] -= local_f[k];
        }
    }

    // 将本线程所有私有寄存器的结果，写入共享内存自己的槽位中
    if (threadIdx.x < 32) {
        for (int k = 0; k < 9; ++k) s_H_all[threadIdx.x][k] = my_H[k];
        for (int k = 0; k < 3; ++k) s_f_all[threadIdx.x][k] = my_f[k];
    }

    // 等待同一 Block 内的所有线程完成所有四面体的计算及写回
    __syncthreads();

    // 线程0最后汇总求逆，更新中心顶点坐标
    if (threadIdx.x == 0) {
        Real final_H[9] = { 0.0f };
        Real final_f[3] = { 0.0f };

        // 1. 先初始化对角海森和惯性力，加上正则化项防止奇异
        Real m = d_masses[v_id];
        Real coeff = m * dtSqrReciprocal;

        // 强制转换常量以适配可能得 double / float (根据你的 Real 定义)
        final_H[0] = coeff + 1e-6f;
        final_H[4] = coeff + 1e-6f;
        final_H[8] = coeff + 1e-6f;

        final_f[0] = coeff * (d_inertia[v_id * 3 + 0] - d_positions[v_id * 3 + 0]);
        final_f[1] = coeff * (d_inertia[v_id * 3 + 1] - d_positions[v_id * 3 + 1]);
        final_f[2] = coeff * (d_inertia[v_id * 3 + 2] - d_positions[v_id * 3 + 2]);

        // 2. 归约汇总：安全地把 Block 内有工作的线程的结果全加起来
        for (int t = 0; t < blockDim.x; ++t) {
            for (int k = 0; k < 9; ++k) final_H[k] += s_H_all[t][k];
            for (int k = 0; k < 3; ++k) final_f[k] += s_f_all[t][k];
        }

        // 3. 最后求解并更新坐标
        Real dx[3];
        if (solve3x3(final_H, final_f, dx)) {
            d_positions[v_id * 3 + 0] += dx[0];
            d_positions[v_id * 3 + 1] += dx[1];
            d_positions[v_id * 3 + 2] += dx[2];
        }
    }
}

namespace VBD {

    // 【新增】步骤B：计算 omega 的辅助函数（与CPU端完全一致）
    static Real getAcceleratorOmega(int order, Real rho, Real prevOmega) {
        if (order == 1) return 1.0f;
        if (order == 2) return 2.0f / (2.0f - (rho * rho));
        return 4.0f / (4.0f - (rho * rho) * prevOmega);
    }

    void VBD_GPU_Solver::initFromCPU(const TetMesh& mesh, const std::vector<std::vector<int>>& vertexColors,
        const std::vector<Matrix3r>& tetDmInv, const std::vector<Real>& tetRestVol) {
        numVerts = mesh.numVerts;
        int numTets = mesh.numTets;

        // 展平并分配显存
        cudaMalloc(&d_positions, numVerts * 3 * sizeof(Real));
        cudaMalloc(&d_inertia, numVerts * 3 * sizeof(Real));
        cudaMalloc(&d_masses, numVerts * sizeof(Real));
        cudaMalloc(&d_is_fixed, numVerts * sizeof(bool));

        cudaMemcpy(d_positions, mesh.mVertPos.data(), numVerts * 3 * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inertia, mesh.mInertia.data(), numVerts * 3 * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_masses, mesh.vertexMass.data(), numVerts * sizeof(Real), cudaMemcpyHostToDevice);

        // 修正 std::vector<bool> 无法获取底层指针的问题
        // 手动提取真实的连续 bool 数组
        bool* host_is_fixed = new bool[numVerts];
        for (int i = 0; i < numVerts; ++i) {
            // 解包 std::vector<bool>
            host_is_fixed[i] = mesh.isFixed[i];
        }
        cudaMemcpy(d_is_fixed, host_is_fixed, numVerts * sizeof(bool), cudaMemcpyHostToDevice);
        delete[] host_is_fixed;

        // 展平四面体
        std::vector<int> flat_tets(numTets * 4);
        for (int i = 0; i < numTets; ++i) for (int j = 0; j < 4; ++j) flat_tets[i * 4 + j] = mesh.tets[i][j];
        cudaMalloc(&d_tets, numTets * 4 * sizeof(int));
        cudaMemcpy(d_tets, flat_tets.data(), numTets * 4 * sizeof(int), cudaMemcpyHostToDevice);

        std::vector<Real> flat_DmInv(numTets * 9);
        for (int i = 0; i < numTets; ++i) {
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) flat_DmInv[i * 9 + a * 3 + b] = tetDmInv[i](a, b);
        }
        cudaMalloc(&d_tetDmInv, numTets * 9 * sizeof(Real));
        cudaMemcpy(d_tetDmInv, flat_DmInv.data(), numTets * 9 * sizeof(Real), cudaMemcpyHostToDevice);

        cudaMalloc(&d_tetRestVol, numTets * sizeof(Real));
        cudaMemcpy(d_tetRestVol, tetRestVol.data(), numTets * sizeof(Real), cudaMemcpyHostToDevice);

        // CSR 格式构建顶点邻接关系
        std::vector<int> adj_offsets(numVerts + 1, 0);
        std::vector<int> adj_data;
        for (int i = 0; i < numVerts; ++i) {
            adj_offsets[i] = adj_data.size();
            adj_data.insert(adj_data.end(), mesh.vertAdjacentTets[i].begin(), mesh.vertAdjacentTets[i].end());
        }
        adj_offsets[numVerts] = adj_data.size();

        cudaMalloc(&d_adj_offsets, (numVerts + 1) * sizeof(int));
        cudaMemcpy(d_adj_offsets, adj_offsets.data(), (numVerts + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_adj_data, adj_data.size() * sizeof(int));
        cudaMemcpy(d_adj_data, adj_data.data(), adj_data.size() * sizeof(int), cudaMemcpyHostToDevice);

        // 按颜色组上传顶点列表
        for (const auto& color : vertexColors) {
            ColorGroup group;
            group.num_vertices = color.size();
            cudaMalloc(&group.d_vertices, group.num_vertices * sizeof(int));
            cudaMemcpy(group.d_vertices, color.data(), group.num_vertices * sizeof(int), cudaMemcpyHostToDevice);
            colorGroups.push_back(group);
        }

        // 【修改】步骤C：在 initFromCPU 底部补充新缓冲区的分配
        cudaMalloc(&d_prevPrevPos, numVerts * 3 * sizeof(Real));
        cudaMalloc(&d_currentPos, numVerts * 3 * sizeof(Real));
        cudaMemset(d_prevPrevPos, 0, numVerts * 3 * sizeof(Real));
        cudaMemset(d_currentPos, 0, numVerts * 3 * sizeof(Real));
    }

    void VBD_GPU_Solver::solveIteration(Real dtSqrReciprocal, Real mu, Real lambda) {
        for (const auto& group : colorGroups) {
            if (group.num_vertices == 0) continue;
            int num_blocks = group.num_vertices;
            int threads_per_block = 16;

            vbd_solve_color_kernel << <num_blocks, threads_per_block >> > (
                group.d_vertices, group.num_vertices, d_positions, d_inertia, d_masses,
                d_adj_offsets, d_adj_data, d_tets, d_tetDmInv, d_tetRestVol, d_is_fixed,
                mu, lambda, dtSqrReciprocal
                );
        }
    }

    // 【新增】步骤E：完整的带有 Chebyshev 的迭代函数
    void VBD_GPU_Solver::solveVBD(int numIterations, Real dtSqrReciprocal, Real mu, Real lambda, bool useAcceleration, Real rho) {
        Real omega = 1.0f;

        for (int iter = 0; iter < numIterations; ++iter) {

            // 1. 保存当前位置 (相当于 CPU 中的 currentPos = mVertPos)
            if (useAcceleration) {
                cudaMemcpy(d_currentPos, d_positions, numVerts * 3 * sizeof(Real), cudaMemcpyDeviceToDevice);
            }

            // 2. 执行核心 Gauss-Seidel 并行求解
            solveIteration(dtSqrReciprocal, mu, lambda);

            // 3. Chebyshev 加速 (完全在显存中操作)
            if (useAcceleration) {
                omega = getAcceleratorOmega(iter + 1, rho, omega);

                if (omega > 1.0f) {
                    int threads = 256;
                    int blocks = (numVerts + threads - 1) / threads;
                    vbd_apply_chebyshev_kernel << <blocks, threads >> > (d_positions, d_prevPrevPos, numVerts, omega, d_is_fixed);
                }

                // 4. 更新历史位置 (相当于 CPU 中的 prevPrevPos = currentPos)
                cudaMemcpy(d_prevPrevPos, d_currentPos, numVerts * 3 * sizeof(Real), cudaMemcpyDeviceToDevice);
            }
        }
    }

    void VBD_GPU_Solver::copyPositionsToCPU(TetMesh& mesh) {
        cudaMemcpy(mesh.mVertPos.data(), d_positions, numVerts * 3 * sizeof(Real), cudaMemcpyDeviceToHost);
    }

    // 【新增】每当 CPU 更新了惯性位置，必须调用此函数上传给 GPU
    void VBD_GPU_Solver::updateInertiaFromCPU(const TetMesh& mesh) {
        cudaMemcpy(d_inertia, mesh.mInertia.data(), numVerts * 3 * sizeof(Real), cudaMemcpyHostToDevice);
    }

    VBD_GPU_Solver::~VBD_GPU_Solver() {
        cudaFree(d_positions); cudaFree(d_inertia); cudaFree(d_masses);
        cudaFree(d_tets); cudaFree(d_tetDmInv); cudaFree(d_tetRestVol);
        cudaFree(d_is_fixed); cudaFree(d_adj_offsets); cudaFree(d_adj_data);
        for (auto& g : colorGroups) cudaFree(g.d_vertices);

        // 【修改】步骤D：记得释放新加的显存
        cudaFree(d_prevPrevPos);
        cudaFree(d_currentPos);
    }

}
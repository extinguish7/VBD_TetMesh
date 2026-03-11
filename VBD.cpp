#include "VBD.h"
#include "GraphColoring.h"
#include "Energy.h"
#include "NeoHookean.h"
#include <omp.h>
#include <cmath>

// 在文件顶部添加全局计时器定义

namespace VBD {
	//PerfTimer g_perfTimer;

    VBDsolver::VBDsolver() {
    }

    VBDsolver::~VBDsolver() {
    }

    void VBDsolver::setParams(const VBDParams& params) {
        this->params = params;
    }

    void VBDsolver::initializeColoring(const TetMesh& mesh) {
        vertexColors = GraphColoring::colorVertices(
            mesh.vertAdjacentTets,
            mesh.tets,
            mesh.numVerts
        );

        prevPrevPos.resize(3, mesh.numVerts);
        prevPrevPos.setZero();

        VBD_LOG("Initialized vertex coloring with " << vertexColors.size() << " color groups");
    }

    void VBDsolver::precomputeTetData(const TetMesh& mesh) {
        tetDmInv.resize(mesh.numTets);
        tetRestVolume.resize(mesh.numTets);

#pragma omp parallel for num_threads(params.numThreads)
        for (int tetId = 0; tetId < mesh.numTets; ++tetId) {
            const auto& tet = mesh.tets[tetId];

            // 计算 Dm (参考配置下的边向量)
            Matrix3r Dm;
            ElasticEnergy::compute_Dm(Dm, mesh.mRestPos, tet);

            // 计算 Dm 的逆
            tetDmInv[tetId] = Dm.inverse();

            // 计算静止体积
            Real volume;
            ElasticEnergy::compute_Volume(volume, Dm);
            tetRestVolume[tetId] = volume;
        }

        VBD_LOG("Precomputed tetrahedron data for " << mesh.numTets << " tets");
    }

    void VBDsolver::step(TetMesh& mesh) {
        // 第一次调用时初始化
        if (vertexColors.empty()) {
            initializeColoring(mesh);
            precomputeTetData(mesh);
        }

        // 子步循环
        Real subDt = params.dt / params.substeps;

        for (int step = 0; step < params.substeps; ++step) {
            // 1. 前向步（计算惯性项）
            forwardStep(mesh);

            // 2. VBD 迭代求解
            Real omega = 1.0;
            for (int iter = 0; iter < params.numIterations; ++iter) {
                // 保存当前位置用于加速
                TVerticesMat currentPos = mesh.mVertPos;

                // 核心求解
                solve(mesh);

                // 3. Chebyshev 加速
                if (params.useAcceleration) {
                    omega = getAcceleratorOmega(iter + 1, params.accelerationRho, omega);
                    if (omega > 1.0) {
                        applyAcceleration(mesh, omega);
                    }
                    prevPrevPos = currentPos;
                }

                // 记录能量（每 10 次迭代）
                if (iter % 10 == 0) {
                    energyHistory.push_back(computeTotalEnergy(mesh));
                }
            }

            // 4. 更新速度
            mesh.updateVelocity(subDt);
        }
    }

    void VBDsolver::forwardStep(TetMesh& mesh) {
        // 保存当前位置
        mesh.savePreviousPosition();

        // 应用重力
        mesh.applyGravity(params.gravity, params.dt / params.substeps);

        // 计算惯性目标位置
        mesh.computeInertia(params.dt / params.substeps);
    }

    void VBDsolver::solve(TetMesh& mesh) {

        Real dt = params.dt / params.substeps;
        Real dtSqrReciprocal = 1.0 / (dt * dt);

        // 按颜色组并行处理
#pragma omp parallel for num_threads(params.numThreads) schedule(dynamic)
        for (int colorIdx = 0; colorIdx < vertexColors.size(); ++colorIdx) {
            const auto& colorGroup = vertexColors[colorIdx];

            for (IdType vertexId : colorGroup) {
                // 跳过固定顶点
                if (mesh.isFixed[vertexId]) {
                    continue;
                }

                Mat3 H = Mat3::Zero();
                Vec3 f = Vec3::Zero();

                //g_perfTimer.start("solve_vertex");
                // 求解单个顶点
                solveVertex(mesh, vertexId, H, f, dtSqrReciprocal);
                //g_perfTimer.end(); // solve_vertex

                // 求解局部线性系统 H * dx = f
                // 检查 H 是否奇异
                //g_perfTimer.start("solve_3x3_system");
                Real detH = H.determinant();
                if (std::abs(detH) > EPSILON) {

                    Vec3 dx;
                    bool success = AnalyticSolver3x3::solve(H, f, dx);
                    // 或者带正则化版本（更稳定）
                    // bool success = AnalyticSolver3x3::solveWithRegularization(H, f, dx, 1e-6f);

                    //Vec3 dx = H.colPivHouseholderQr().solve(f);

                    // 更新顶点位置
                    mesh.vertex(vertexId) += dx;
                }
                //g_perfTimer.end(); // solve_3x3_system
                // 如果 H 奇异，跳过该顶点（论文 Section 3.2）
            }
        }
    }

    void VBDsolver::solveVertex(
        const TetMesh& mesh,
        IdType vertexId,
        Mat3& H,
        Vec3& f,
        Real dtSqrReciprocal
    ) {
        // 1. 惯性项 (论文 Eq 8, 9)
        // H_inertia = m_i / h^2 * I
        // f_inertia = m_i / h^2 * (y_i - x_i)
        H += mesh.vertexMass(vertexId) * dtSqrReciprocal * Mat3::Identity();
        f = mesh.vertexMass(vertexId) *
            (mesh.mInertia.col(vertexId) - mesh.vertex(vertexId)) *
            dtSqrReciprocal;

        // 2. Neo-Hookean 弹性力项
        // 遍历该顶点相邻的所有四面体
        for (IdType tetId : mesh.vertAdjacentTets[vertexId]) {
            // 找到顶点在四面体中的局部索引 (0-3)
            int localIdx = -1;
            const auto& tet = mesh.tets[tetId];
            for (int i = 0; i < 4; ++i) {
                if (tet[i] == vertexId) {
                    localIdx = i;
                    break;
                }
            }

            if (localIdx == -1) continue;

            Mat3 H_local = Mat3::Zero();
            Vec3 f_local = Vec3::Zero();

            computeTetContribution(mesh, tetId, vertexId, localIdx, H_local, f_local);

			/*std::cout << "Tet " << tetId << ", Vertex " << vertexId
				<< ", LocalIdx " << localIdx
				<< ", H_local:\n" << H_local
				<< ", f_local: " << f_local.transpose() << std::endl;*/

            H += H_local;
            f -= f_local;
        }
    }

    void VBDsolver::computeTetContribution(
        const TetMesh& mesh,
        IdType tetId,
        IdType vertexId,
        int localVertexIdx,
        Mat3& H_local,
        Vec3& f_local
    ) {
        const auto& tet = mesh.tets[tetId];
        Real mu = params.mu;
        Real lambda = params.lambda;
        Real V0 = tetRestVolume[tetId];
        const Matrix3r& DmInv = tetDmInv[tetId];

        // 材料参数向量
        std::vector<Real> material_params = { mu, lambda };

        //g_perfTimer.start("solve_Dm");

        // 1. 计算当前配置下的 Ds 矩阵
        Matrix3r Ds;
        ElasticEnergy::compute_Dm(Ds, mesh.mVertPos, tet);
        //g_perfTimer.end(); // solve_Dm

        //g_perfTimer.start("solve_F");
        // 2. 计算变形梯度 F = Ds * DmInv
        Matrix3r F;
        ElasticEnergy::compute_F(F, Ds, DmInv);
        //g_perfTimer.end(); // solve_F

        //g_perfTimer.start("solve_Hessian_and_Gradient");
        Matrix12r eleHessian;
        Vec12 eleGradient;
        ElasticEnergy::compute_Hessian_and_Gradient(
            eleHessian,
            eleGradient,
            Ds, 
            DmInv, 
            F, 
            lambda, 
            mu, 
			V0);
        //g_perfTimer.end();


        // 6. 提取该顶点的梯度 (3 维)
        int startIdx = localVertexIdx * 3;
        f_local << eleGradient[startIdx], eleGradient[startIdx + 1], eleGradient[startIdx + 2];

        // 7. 提取该顶点的海森块 (3x3)
        H_local = eleHessian.block(startIdx, startIdx, 3, 3);
    }

    Real VBDsolver::computeTotalEnergy(const TetMesh& mesh) const {
        Real totalEnergy = 0.0;

        // 弹性势能
        for (size_t tetId = 0; tetId < mesh.numTets; ++tetId) {
            const auto& tet = mesh.tets[tetId];
            Real mu = params.mu;
            Real lambda = params.lambda;
            Real V0 = tetRestVolume[tetId];
            const Matrix3r& DmInv = tetDmInv[tetId];

            // 计算当前 F
            Matrix3r Ds;
            ElasticEnergy::compute_Dm(Ds, mesh.mVertPos, tet);
            Matrix3r F = Ds * DmInv;

            // SVD
            Matrix3r U, V;
            Vec3 sigma;
            NeoHookean::compute_SVD(U, sigma, V, F);

            // 能量
            std::vector<Real> material_params = { mu, lambda };
            Real E;
            NeoHookean::compute_Energy(E, F, sigma, U, V, material_params);

            totalEnergy += E * V0;
        }

        return totalEnergy;
    }

    void VBDsolver::applyAcceleration(TetMesh& mesh, Real omega) {
        if (omega <= 1.0) return;

#pragma omp parallel for num_threads(params.numThreads)
        for (int i = 0; i < mesh.numVerts; ++i) {
            if (!mesh.isFixed[i]) {
                mesh.mVertPos.col(i) = omega * (mesh.mVertPos.col(i) - prevPrevPos.col(i)) +
                    prevPrevPos.col(i);
            }
        }
    }

    Real VBDsolver::getAcceleratorOmega(int order, Real rho, Real prevOmega) {
        switch (order) {
        case 1:
            return 1.0;
        case 2:
            return 2.0 / (2.0 - SQR(rho));
        default:
            return 4.0 / (4.0 - SQR(rho) * prevOmega);
        }
    }

    void VBDsolver::simulate(TetMesh& mesh, int numFrames,
        std::function<void(int, TetMesh&)> callback) {
        energyHistory.clear();
        //g_perfTimer.reset();

        for (int frame = 0; frame < numFrames; ++frame) {
            step(mesh);

            Real energy = computeTotalEnergy(mesh);
            VBD_LOG("Frame " << frame << " completed, Energy: " << energy);

            if (callback) {
                callback(frame, mesh);
            }
        }
        // 打印详细性能报告
        //g_perfTimer.printReport();
    }

} // namespace VBD
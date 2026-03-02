#include "Types.h"
#include "TetMesh.h"
#include "VBD.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include "VBD_GPU_Wrapper.h"

using namespace VBD;
using namespace std;

// 【修改1：将 density, mu, lambda 作为参数传入，保证与 main 函数中的参数绝对一致】
void loadTetgenModel(const std::string& nodeFilename, const std::string& eleFilename, TetMesh& mesh, Real density, Real mu, Real lambda)
{
    cout << "Loading " << nodeFilename << endl;
    cout << "Loading " << eleFilename << endl;

    std::vector<Vec3> positions;
    std::vector<TetVIdsArr> tets;

    // variables
    size_t i, num_vertices, num_tetras;
    string nodeLine, eleLine, label;
    stringstream sStream;
    // try to open the file
    ifstream finNode(nodeFilename.c_str());
    ifstream finEle(eleFilename.c_str());
    if (!finNode)
    {
        cout << "'" + nodeFilename + "' file not found." << endl;
        return;
    }
    if (!finEle)
    {
        cout << "'" + eleFilename + "' file not found." << endl;
        return;
    }

    // get num vertices
    getline(finNode, nodeLine);
    sStream << nodeLine;
    sStream >> num_vertices;
    sStream >> label; // 3
    sStream >> label; // 0
    sStream >> label; // 0
    sStream.clear();

    // get num tetras
    getline(finEle, eleLine);
    sStream << eleLine;
    sStream >> num_tetras;
    sStream >> label; // 4
    sStream >> label; // 0
    sStream >> label; // 0
    sStream.clear();

    positions.resize(num_vertices);
    tets.resize(num_tetras);

    bool start_from_0 = false;

    // read vertices
    for (i = 0; i < num_vertices; ++i)
    {
        unsigned nodeInd;
        Real x, y, z;
        getline(finNode, nodeLine);
        sStream << nodeLine;
        sStream >> nodeInd >> x >> y >> z;
        if (nodeInd == 0)
            start_from_0 = true;

        getline(sStream, nodeLine);
        sStream.clear();

        positions[i] = Vec3(x, y, z);
    }

    // read tetrahedra
    for (i = 0; i < num_tetras; ++i)
    {
        unsigned eleInd;
        getline(finEle, eleLine);
        sStream << eleLine;

        sStream >> eleInd >> tets[i][0] >> tets[i][1] >> tets[i][2] >> tets[i][3];

        if (!start_from_0)
        {
            tets[i][0] -= 1; tets[i][1] -= 1; tets[i][2] -= 1; tets[i][3] -= 1;
        }

        getline(sStream, eleLine);
        sStream.clear();
    }
    // close file
    finNode.close();
    finEle.close();

    cout << "Number of tets: " << num_tetras << endl;
    cout << "Number of vertices: " << num_vertices << endl;

    // 【修改2：移除原来的 masses 数组】
    // std::vector<Real> masses(positions.size(), 0.1);

    // 固定左端
    // 1. 计算所有顶点的 X 轴包围盒边界
    FloatingType min_x = std::numeric_limits<FloatingType>::max();
    FloatingType max_x = std::numeric_limits<FloatingType>::lowest();

    for (const auto& pos : positions) {
        if (pos.x() < min_x) min_x = pos.x();
        if (pos.x() > max_x) max_x = pos.x();
    }

    // 2. 计算最左侧 1/10 区域的阈值 (注意这里的 0.1f 强制要求单精度)
    FloatingType threshold_x = min_x + (max_x - min_x) * 0.1f;

    // 3. 收集该区域内的所有顶点索引作为固定顶点
    std::vector<IdType> fixedVerts;
    for (size_t i = 0; i < positions.size(); ++i) {
        // 如果顶点的 x 坐标小于等于阈值，则将其固定
        if (positions[i].x() <= threshold_x) {
            fixedVerts.push_back(static_cast<IdType>(i));
        }
    }

    // 【修改3：调用最新的 initialize，传入 density】
    mesh.initialize(positions, density, tets, mu, lambda, fixedVerts);
}

int main() {
    // 【修改4：将物理参数提前计算，以传给 Mesh 加载器保证矩阵初始化正确】
    Real youngs_modulus = 100000.0e-6f; // 加上 f 后缀
    Real poisson_ratio = 0.3f;
    Real mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    Real lambda = youngs_modulus * poisson_ratio / ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));
    Real density = 1000.0e-12f;     // 设置材料密度

    // 1. 初始化你的 Tetrahedral Mesh 模型 
    TetMesh mesh;
    loadTetgenModel("D:/test_max_CCD_time/VBD_TetMesh/armadillo_40k.node", "D:/test_max_CCD_time/VBD_TetMesh/armadillo_40k.ele", mesh, density, mu, lambda);
    mesh.exportToObj("D:/test_max_CCD_time/VBD_TetMesh/output/origin.obj");

    std::cout << "Mesh created: " << mesh.numVerts << " vertices, "
        << mesh.numTets << " tetrahedra" << std::endl;

    VBDParams params;
    params.dt = 0.01f;              // 10ms 时间步长
    params.numIterations = 30;      // 每步 30 次迭代
    params.substeps = 2;            // 2 个子步
    params.gravity << 0.0f, -9.8e3f, 0.0f; // 重力 (加 f 防止隐式转换双精度)
    params.useAcceleration = true;  // 启用 Chebyshev 加速
    params.accelerationRho = 0.75f; // 谱半径估计
    params.numThreads = 4;          // 4 线程并行
    params.mu = mu;                 // Neo-Hookean μ
    params.lambda = lambda;         // Neo-Hookean λ

    // 2. CPU端的预计算与图着色
    VBDsolver cpuSolver;
    cpuSolver.setParams(params);
    cpuSolver.initializeColoring(mesh); // 会自动生成 vertexColors
    cpuSolver.precomputeTetData(mesh);  // 会计算得到 tetDmInv 和 tetRestVol

    // 3. 将计算下发到 CUDA GPU 环境
    std::cout << "Transferring mesh data to GPU..." << std::endl;
    VBD_GPU_Solver gpuSolver;
    gpuSolver.initFromCPU(mesh, cpuSolver.getVertexColors(), cpuSolver.getTetDmInv(), cpuSolver.getTetRestVol());

    Real subDt = params.dt / params.substeps;
    Real dtSqrReciprocal = 1.0f / (subDt * subDt);

    // 4. 开始主模拟循环
    for (int frame = 0; frame < 3000; ++frame) {

        for (int step = 0; step < params.substeps; ++step) {

            // 每次物理子步必须由 CPU 计算新的惯性位置并更新到 GPU
            cpuSolver.forwardStep(mesh);
            
            // 将更新的惯性数据传给GPU
            gpuSolver.updateInertiaFromCPU(mesh);

            gpuSolver.solveVBD(params.numIterations, dtSqrReciprocal, params.mu, params.lambda,
                params.useAcceleration, params.accelerationRho);

            // 将更新后的位置读回 CPU 以便用于速度更新
            gpuSolver.copyPositionsToCPU(mesh);
            mesh.updateVelocity(subDt);
        }

        std::cout << "Frame " << frame << " simulated on GPU." << std::endl;

        if (frame % 100 == 0) {
            std::string filename = "output/frame_" + std::to_string(frame) + ".txt";
            mesh.exportToObj("D:/test_max_CCD_time/VBD_TetMesh/output/frame_" + std::to_string(frame) + ".obj");
        }
    }

    return 0;
}
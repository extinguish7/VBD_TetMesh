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

// 辅助函数：读取 .node 文件
std::vector<Vec3> readNodeFile(const std::string& nodeFilename, bool& start_from_0) {
    std::vector<Vec3> positions;
    ifstream finNode(nodeFilename.c_str());
    if (!finNode) {
        cout << "'" + nodeFilename + "' file not found." << endl;
        return positions;
    }
    size_t num_vertices;
    string nodeLine, label;
    stringstream sStream;

    getline(finNode, nodeLine);
    sStream << nodeLine;
    sStream >> num_vertices >> label >> label >> label;
    positions.resize(num_vertices);

    for (size_t i = 0; i < num_vertices; ++i) {
        sStream.clear();
        unsigned nodeInd;
        Real x, y, z;
        getline(finNode, nodeLine);
        sStream << nodeLine;
        sStream >> nodeInd >> x >> y >> z;

        // 判断顶点编号是否从 0 开始
        if (i == 0 && nodeInd == 0) start_from_0 = true;

        positions[i] = Vec3(x, y, z);
    }
    finNode.close();
    return positions;
}

// 加载模型，支持传入【静止形态节点文件】和【当前形态节点文件】
void loadTetgenModel(const std::string& restNodeFilename, const std::string& currentNodeFilename, const std::string& eleFilename, TetMesh& mesh, Real density, Real mu, Real lambda)
{
    cout << "Loading Rest Node: " << restNodeFilename << endl;
    cout << "Loading Current Node: " << currentNodeFilename << endl;
    cout << "Loading Ele: " << eleFilename << endl;

    bool start_from_0 = false;

    // 1. 读取初始形态位置
    std::vector<Vec3> rest_positions = readNodeFile(restNodeFilename, start_from_0);

    // 2. 读取当前形态位置
    bool dummy_start = false;
    std::vector<Vec3> current_positions = readNodeFile(currentNodeFilename, dummy_start);

    if (rest_positions.empty()) {
        cout << "Failed to read node files!" << endl;
        return;
    }
    else if (current_positions.empty())
	{
		cout << "Failed to read current node file!" << endl;
		current_positions = rest_positions; // 如果没有当前节点文件，则使用静止形态位置作为当前形态位置

    }

    if (rest_positions.size() != current_positions.size()) {
        cout << "Error: Rest nodes and current nodes count mismatch!" << endl;
        return;
    }

    // mm转换为m
	//for (auto& pos : rest_positions) {
	//	pos *= 0.001f; // 将位置从毫米转换为米
	//}
 //   for (auto& pos : current_positions) {
	//	pos *= 0.001f; // 将位置从毫米转换为米
 //   }

    // 3. 读取元素文件
    std::vector<TetVIdsArr> tets;
    size_t num_tetras;
    string eleLine, label;
    stringstream sStream;

    ifstream finEle(eleFilename.c_str());
    if (!finEle) {
        cout << "'" + eleFilename + "' file not found." << endl;
        return;
    }

    getline(finEle, eleLine);
    sStream << eleLine;
    sStream >> num_tetras >> label >> label >> label;
    sStream.clear();

    tets.resize(num_tetras);

    for (size_t i = 0; i < num_tetras; ++i) {
        unsigned eleInd;
        getline(finEle, eleLine);
        sStream << eleLine;
        sStream >> eleInd >> tets[i][0] >> tets[i][1] >> tets[i][2] >> tets[i][3];

        if (!start_from_0) {
            tets[i][0] -= 1; tets[i][1] -= 1; tets[i][2] -= 1; tets[i][3] -= 1;
        }
        sStream.clear();
    }
    finEle.close();

    cout << "Number of tets: " << num_tetras << endl;
    cout << "Number of vertices: " << rest_positions.size() << endl;

    // 4. 固定左端边界条件 (必须基于没有变形的 rest_positions 包围盒来进行筛选)

    std::vector<IdType> fixedVerts;
    bool fixed_X = false;
    bool fixed_Y = true;
    if (fixed_X)
    {
		FloatingType min_x = std::numeric_limits<FloatingType>::max();
		FloatingType max_x = std::numeric_limits<FloatingType>::lowest();

        for (const auto& pos : current_positions) {
            if (pos.x() < min_x) min_x = pos.x();
            if (pos.x() > max_x) max_x = pos.x();
        }

        FloatingType threshold_x = min_x + (max_x - min_x) * 0.05f;

        for (size_t i = 0; i < current_positions.size(); ++i) {
            // 如果顶点的静止 x 坐标小于等于阈值，则将其固定
            if (current_positions[i].x() <= threshold_x) {
                fixedVerts.push_back(static_cast<IdType>(i));
            }
        }
    }
	if (fixed_Y)
	{
		FloatingType min_y = std::numeric_limits<FloatingType>::max();
		FloatingType max_y = std::numeric_limits<FloatingType>::lowest();

		for (const auto& pos : current_positions) {
			if (pos.y() < min_y) min_y = pos.y();
			if (pos.y() > max_y) max_y = pos.y();
		}

		FloatingType threshold_y = min_y + (max_y - min_y) * 0.05f;

		for (size_t i = 0; i < current_positions.size(); ++i) {
			// 如果顶点的静止 y 坐标小于等于阈值，则将其固定
			if (current_positions[i].y() <= threshold_y) {
				fixedVerts.push_back(static_cast<IdType>(i));
			}
		}
	}

    // 5. 初始化 Mesh (传入双重 positions)
    mesh.initialize(rest_positions, current_positions, density, tets, mu, lambda, fixedVerts);
}

int main() {
    // 【修改4：将物理参数提前计算，以传给 Mesh 加载器保证矩阵初始化正确】
    Real youngs_modulus = 200.0e3f; // 加上 f 后缀
    Real poisson_ratio = 0.3f;
    Real mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    Real lambda = youngs_modulus * poisson_ratio / ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));
    Real density = 7800.0e-12f;     // 设置材料密度

    // 1. 初始化你的 Tetrahedral Mesh 模型 
    TetMesh mesh;
    //loadTetgenModel("D:/test_max_CCD_time/VBD_TetMesh/tet1800.node", "", "D:/test_max_CCD_time/VBD_TetMesh/tet1800.ele", mesh, density, mu, lambda);
    //loadTetgenModel("D:/test_max_CCD_time/VBD_TetMesh/armadillo_40k.node","", "D:/test_max_CCD_time/VBD_TetMesh/armadillo_40k.ele", mesh, density, mu, lambda);
    loadTetgenModel("D:/test_max_CCD_time/VBD_TetMesh/VenusA26_Y.node", "D:/test_max_CCD_time/VBD_TetMesh/VenusA26_compressed_D8mm_Y.node", "D:/test_max_CCD_time/VBD_TetMesh/VenusA26_Y.ele", mesh, density, mu, lambda);
    mesh.exportToObj("D:/test_max_CCD_time/VBD_TetMesh/output/origin.obj");

    std::cout << "Mesh created: " << mesh.numVerts << " vertices, "
        << mesh.numTets << " tetrahedra" << std::endl;

    VBDParams params;
    params.dt = 0.001f;              // 10ms 时间步长
    params.numIterations = 50;      // 每步 30 次迭代
    params.substeps = 2;            // 2 个子步
    params.gravity << 0.0f, 0.0f, 0.0f; // 重力 -9.8f(加 f 防止隐式转换双精度)
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
	int numFrames = 3000;
    for (int frame = 0; frame < numFrames; ++frame) {

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

		if (frame % 30 == 0 || frame == numFrames - 1) {
            std::string filename = "output/frame_" + std::to_string(frame) + ".txt";
            mesh.exportToObj("D:/test_max_CCD_time/VBD_TetMesh/output/frame_" + std::to_string(frame) + ".obj");
        }
    }

    return 0;
}


//int main(int argc, char** argv) {
//    std::cout << "=== VBD TetMesh Neo-Hookean Simulator ===" << std::endl;
//    std::cout << "Using full Neo-Hookean Hessian from your code" << std::endl;
//
//    // 创建求解器
//    VBDsolver solver;
//
//    // 设置参数
//    Real youngs_modulus = 1e5f; // 加上 f 后缀
//    Real poisson_ratio = 0.3f;
//    Real mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
//    Real lambda = youngs_modulus * poisson_ratio / ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));
//    Real density = 1000.0f;     // 设置材料密度
//
//    VBDParams params;
//    params.dt = 0.01;              // 10ms 时间步长
//    params.numIterations = 50;     // 每步 50 次迭代
//    params.substeps = 2;           // 2 个子步
//    params.gravity << 0, -9.8, 0;  // 重力
//    params.useAcceleration = true; // 启用 Chebyshev 加速
//    params.accelerationRho = 0.75; // 谱半径估计
//    params.numThreads = 4;         // 4 线程并行
//    params.mu = mu;                                                // Neo-Hookean μ
//    params.lambda = lambda;    // Neo-Hookean λ
//
//    solver.setParams(params);
//
//    // 创建网格
//    TetMesh mesh;
//	loadTetgenModel("D:/test_max_CCD_time/VBD_TetMesh/armadillo_4k.node","", "D:/test_max_CCD_time/VBD_TetMesh/armadillo_4k.ele", mesh, density, mu, lambda);
//    mesh.exportToObj("D:/test_max_CCD_time/VBD_TetMesh/output/origin.obj");
//
//
//    std::cout << "Mesh created: " << mesh.numVerts << " vertices, "
//        << mesh.numTets << " tetrahedra" << std::endl;
//
//    // 模拟
//    int numFrames = 300;
//    std::cout << "Starting simulation for " << numFrames << " frames..." << std::endl;
//
//    auto startTime = std::chrono::high_resolution_clock::now();
//
//    solver.simulate(mesh, numFrames, [&](int frame, TetMesh& m) {
//        // 每 10 帧保存一次
//        if (frame % 10 == 0) {
//            std::string filename = "output/frame_" + std::to_string(frame) + ".txt";
//
//            m.exportToObj("D:/test_max_CCD_time/VBD_TetMesh/output/frame_"+std::to_string(frame)+".obj");
//
//            Real energy = solver.computeTotalEnergy(m);
//            std::cout << "Frame " << frame << ": Energy = " << energy << std::endl;
//        }
//        });
//
//    auto endTime = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//
//    std::cout << "\n=== Simulation Complete ===" << std::endl;
//    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
//    std::cout << "Average time per frame: " << (duration.count() / static_cast<Real>(numFrames)) << " ms" << std::endl;
//    std::cout << "Color groups: " << solver.getVertexColors().size() << std::endl;
//
//    return 0;
//}
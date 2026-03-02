#pragma once

#include "Types.h"
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <unordered_map>

namespace VBD {

    /**
     * @brief 四面体网格数据结构
     *
     * 存储顶点位置、速度、质量以及四面体单元信息
     */
    struct TetMesh {
        // 顶点数据
        TVerticesMat mVertPos;          // 当前位置 (3 x N)
        TVerticesMat mVertPrevPos;      // 上一帧位置 (3 x N)
        TVerticesMat mVelocity;         // 速度 (3 x N)
        TVerticesMat mInertia;          // 惯性目标位置 (3 x N)
        VecDynamic vertexMass;          // 顶点质量 (N)

        // 四面体单元
        std::vector<TetVIdsArr> tets;           // 四面体顶点索引
        std::vector<FloatingType> tetRestVol;   // 四面体静止体积
        std::vector<FloatingType> tetMu;        // 四面体材料参数 μ
        std::vector<FloatingType> tetLambda;    // 四面体材料参数 λ

        // 邻接关系
        std::vector<std::vector<IdType>> vertAdjacentTets;  // 顶点相邻的四面体

        // 顶点数量
        size_t numVerts;
        size_t numTets;

        // 固定顶点标记
        std::vector<bool> isFixed;

        // 三角形面（排序后的顶点索引，用于唯一标识）
        struct TriangleFace {
            IdType v[3];

            TriangleFace() {
                v[0] = v[1] = v[2] = -1;
            }

            TriangleFace(IdType a, IdType b, IdType c) {
                v[0] = a; v[1] = b; v[2] = c;
                // 排序以便识别相同的面（无论顶点顺序）
                std::sort(v, v + 3);
            }

            bool operator==(const TriangleFace& other) const {
                return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
            }

            bool operator<(const TriangleFace& other) const {
                if (v[0] != other.v[0]) return v[0] < other.v[0];
                if (v[1] != other.v[1]) return v[1] < other.v[1];
                return v[2] < other.v[2];
            }
        };

        // 带方向的三角形（记录原始顶点顺序，用于法线方向）
        struct OrientedTriangle {
            IdType v0, v1, v2;
			OrientedTriangle() : v0(-1), v1(-1), v2(-1) {}
            OrientedTriangle(IdType a, IdType b, IdType c) : v0(a), v1(b), v2(c) {}
        };

        // ============================================================================
// 表面提取实现
// ============================================================================

        std::vector<Vec3I> extractSurfaceFaces() const {
            std::vector<Vec3I> surfaceFaces;

            if (numTets == 0 || tets.empty()) {
                return surfaceFaces;
            }

            // 使用 map 统计每个面出现的次数（map 自动处理比较，不需要自定义哈希）
            // key: 排序后的三角形（唯一标识）
            // value: 出现次数
            std::map<TriangleFace, int> faceCount;

            // 记录每个面的原始方向（第一次出现时的顶点顺序）
            std::map<TriangleFace, OrientedTriangle> faceOrientation;

            // 遍历所有四面体
            for (size_t tetIdx = 0; tetIdx < numTets; ++tetIdx) {
                const auto& tet = tets[tetIdx];
                IdType v0 = tet[0];
                IdType v1 = tet[1];
                IdType v2 = tet[2];
                IdType v3 = tet[3];

                // 四面体的4个三角形面
                // 顶点顺序：确保每个面的法线指向四面体外侧
                // 对于四面体 (v0,v1,v2,v3)，外法线方向如下：
                OrientedTriangle faces[4] = {
                    OrientedTriangle(v0, v2, v1),  // 对面 v3
                    OrientedTriangle(v0, v1, v3),  // 对面 v2
                    OrientedTriangle(v0, v3, v2),  // 对面 v1
                    OrientedTriangle(v1, v2, v3)   // 对面 v0
                };

                for (int f = 0; f < 4; ++f) {
                    const OrientedTriangle& tri = faces[f];
                    TriangleFace key(tri.v0, tri.v1, tri.v2);

                    if (faceCount.find(key) == faceCount.end()) {
                        faceCount[key] = 1;
                        faceOrientation[key] = tri;
                    }
                    else {
                        faceCount[key]++;
                    }
                }
            }

            // 提取只出现1次的面（边界表面）
            for (std::map<TriangleFace, int>::const_iterator it = faceCount.begin();
                it != faceCount.end(); ++it) {
                if (it->second == 1) {
                    const TriangleFace& key = it->first;
                    std::map<TriangleFace, OrientedTriangle>::const_iterator oriIt = faceOrientation.find(key);
                    if (oriIt != faceOrientation.end()) {
                        const OrientedTriangle& tri = oriIt->second;
                        surfaceFaces.push_back(Vec3I(tri.v0, tri.v1, tri.v2));
                    }
                }
            }

            return surfaceFaces;
        }

        // ============================================================================
        // OBJ 导出实现
        // ============================================================================

        bool exportToObj(const std::string& filepath) const {
            std::ofstream outFile(filepath.c_str());

            if (!outFile.is_open()) {
                std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
                return false;
            }

            // 写入 OBJ 文件头
            outFile << "# TetMesh OBJ Export\n";
            outFile << "# Vertices: " << numVerts << "\n";
            outFile << "# Tetrahedra: " << numTets << "\n\n";

            // 1. 写入顶点 (v x y z)
            for (size_t i = 0; i < numVerts; ++i) {
                Vec3 pos = mVertPos.col(i);
                outFile << "v " << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
            }

            outFile << "\n";

            // 2. 提取表面并写入面片 (f v1 v2 v3)
            std::vector<Vec3I> surfaceFaces = extractSurfaceFaces();

            outFile << "# Surface Triangles: " << surfaceFaces.size() << "\n";

            for (size_t i = 0; i < surfaceFaces.size(); ++i) {
                const Vec3I& face = surfaceFaces[i];
                // OBJ 索引从 1 开始
                outFile << "f " << (face[0] + 1) << " " << (face[1] + 1) << " " << (face[2] + 1) << "\n";
            }

            outFile.close();

            std::cout << "Exported TetMesh to OBJ: " << filepath << std::endl;
            std::cout << "  Vertices: " << numVerts << std::endl;
            std::cout << "  Surface Faces: " << surfaceFaces.size() << std::endl;

            return true;
        }

        /**
         * @brief 获取顶点位置的可写块
         */
        Eigen::Block<TVerticesMat, 3, 1> vertex(IdType i) {
            return mVertPos.block<3, 1>(0, i);
        }

        /**
         * @brief 获取顶点位置的只读块
         */
        const Eigen::Block<const TVerticesMat, 3, 1> vertex(IdType i) const {
            return mVertPos.block<3, 1>(0, i);
        }

        /**
         * @brief 初始化网格 (基于密度自动分配质量)
         *
         * @param positions 顶点初始位置
         * @param density 材料密度 (例如水的密度为 1000.0e-12)
         * @param tetIndices 四面体顶点索引
         * @param mu 材料参数 μ
         * @param lambda 材料参数 λ
         * @param fixedVerts 固定顶点索引
         */
        void initialize(
            const std::vector<Vec3>& positions,
            FloatingType density,
            const std::vector<TetVIdsArr>& tetIndices,
            FloatingType mu,
            FloatingType lambda,
            const std::vector<IdType>& fixedVerts = {}
        ) {
            numVerts = positions.size();
            numTets = tetIndices.size();

            // 初始化顶点数据
            mVertPos.resize(3, numVerts);
            mVertPrevPos.resize(3, numVerts);
            mVelocity.resize(3, numVerts);
            mInertia.resize(3, numVerts);
            vertexMass.resize(numVerts);

            mVertPos.setZero();
            mVertPrevPos.setZero();
            mVelocity.setZero();
            vertexMass.setZero(); // <--- 【修改2：先清零，准备后续累加】

            for (size_t i = 0; i < numVerts; ++i) {
                mVertPos.col(i) = positions[i];
                mVertPrevPos.col(i) = positions[i];
            }

            // 初始化四面体
            tets = tetIndices;
            tetRestVol.resize(numTets);
            tetMu.resize(numTets, mu);
            tetLambda.resize(numTets, lambda);
            vertAdjacentTets.resize(numVerts);

            // 【修改3：计算四面体体积，并根据密度将质量均分给4个顶点】
            for (size_t i = 0; i < numTets; ++i) {
                FloatingType vol = computeRestVolume(i);
                tetRestVol[i] = vol;

                // 当前四面体的总质量 = 体积 * 密度
                FloatingType tetMass = vol * density;

                for (int j = 0; j < 4; ++j) {
                    vertAdjacentTets[tets[i][j]].push_back(static_cast<IdType>(i));
                    // 将四面体的质量均分给4个顶点 (使用 += 因为一个顶点可能被多个四面体共享)
                    vertexMass(tets[i][j]) += tetMass / 4.0;
                }
            }

            // 标记固定顶点
            isFixed.assign(numVerts, false);
            for (IdType vid : fixedVerts) {
                if (vid >= 0 && vid < static_cast<IdType>(numVerts)) {
                    isFixed[vid] = true;
                }
            }
        }

        /**
         * @brief 计算四面体体积
         */
        FloatingType computeTetVolume(IdType tetId) const {
            const auto& v = tets[tetId];
            Vec3 e0 = mVertPos.col(v[1]) - mVertPos.col(v[0]);
            Vec3 e1 = mVertPos.col(v[2]) - mVertPos.col(v[0]);
            Vec3 e2 = mVertPos.col(v[3]) - mVertPos.col(v[0]);
            return std::abs(e0.dot(e1.cross(e2))) / 6.0;
        }

        /**
         * @brief 计算四面体静止体积（用于初始化）
         */
        FloatingType computeRestVolume(IdType tetId) const {
            const auto& v = tets[tetId];
            Vec3 e0 = mVertPos.col(v[1]) - mVertPos.col(v[0]);
            Vec3 e1 = mVertPos.col(v[2]) - mVertPos.col(v[0]);
            Vec3 e2 = mVertPos.col(v[3]) - mVertPos.col(v[0]);
            return std::abs(e0.dot(e1.cross(e2))) / 6.0;
        }

        /**
         * @brief 更新速度
         *
         * @param dt 时间步长
         */
        void updateVelocity(FloatingType dt) {
            for (size_t i = 0; i < numVerts; ++i) {
                if (!isFixed[i]) {
                    mVelocity.col(i) = (mVertPos.col(i) - mVertPrevPos.col(i)) / dt;
                }
            }
        }

        /**
         * @brief 应用重力
         *
         * @param gravity 重力加速度
         * @param dt 时间步长
         */
        void applyGravity(const Vec3& gravity, FloatingType dt) {
            for (size_t i = 0; i < numVerts; ++i) {
                if (!isFixed[i]) {
                    mVelocity.col(i) += dt * gravity;
                }
            }
        }

        /**
         * @brief 计算惯性目标位置
         *
         * @param dt 时间步长
         */
        void computeInertia(FloatingType dt) {
            for (size_t i = 0; i < numVerts; ++i) {
                if (!isFixed[i]) {
                    mInertia.col(i) = mVertPos.col(i) + dt * mVelocity.col(i);
                }
                else {
                    mInertia.col(i) = mVertPos.col(i);
                }
            }
        }

        /**
         * @brief 保存当前位置为上一帧
         */
        void savePreviousPosition() {
            mVertPrevPos = mVertPos;
        }
    };

} // namespace VBD
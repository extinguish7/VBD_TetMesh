#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <functional>
#include <chrono>

// 基本类型定义
using FloatingType = float;
using Real = FloatingType;  // 与你的代码兼容

namespace VBD {

	using IdType = int32_t;

	// 向量类型
	using Vec2 = Eigen::Matrix<Real, 2, 1>;
	using Vec3 = Eigen::Matrix<Real, 3, 1>;
	using Vec4 = Eigen::Matrix<Real, 4, 1>;
	using Vec6 = Eigen::Matrix<Real, 6, 1>;
	using Vec9 = Eigen::Matrix<Real, 9, 1>;
	using Vec12 = Eigen::Matrix<Real, 12, 1>;
	using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

	using Vec2I = Eigen::Matrix<IdType, 2, 1>;
	using Vec3I = Eigen::Matrix<IdType, 3, 1>;
	using Vec4I = Eigen::Matrix<IdType, 4, 1>;
	using Eigen::Vector4i;

	// 矩阵类型
	using Mat2 = Eigen::Matrix<Real, 2, 2>;
	using Mat3 = Eigen::Matrix<Real, 3, 3>;
	using Mat4 = Eigen::Matrix<Real, 4, 4>;
	using Mat6 = Eigen::Matrix<Real, 6, 6>;
	using Mat9 = Eigen::Matrix<Real, 9, 9>;
	using Mat12 = Eigen::Matrix<Real, 12, 12>;
	using Mat3x4 = Eigen::Matrix<Real, 3, 4>;
	using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
	using Matrix3r = Mat3;
	using Matrix9r = Mat9;
	using Matrix12r = Mat12;

	// 动态矩阵
	using TVerticesMat = Eigen::Matrix<Real, 3, Eigen::Dynamic>;
	using VecDynamic = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
	using VecDynamicI = Eigen::Matrix<IdType, Eigen::Dynamic, 1>;

	// 四面体顶点索引
	using TetVIdsArr = std::array<IdType, 4>;
	using TTetVIdsArr = TetVIdsArr;

	// 稀疏矩阵
	using SpMat = Eigen::SparseMatrix<Real>;
	using Triplet = Eigen::Triplet<Real>;
	using SPARSE_MATRIX = SpMat;

	// 常用宏
#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
#define EPSILON 1e-8
#define EPSILON2 (EPSILON * EPSILON)

// 调试输出
#ifdef VBD_DEBUG
#define VBD_LOG(msg) std::cout << "[VBD] " << msg << std::endl
#else
#define VBD_LOG(msg)
#endif

	/**
 * @brief 3x3 线性系统解析求解器
 * 使用伴随矩阵法，比 QR 分解快 3-5 倍
 */
	class AnalyticSolver3x3 {
	public:
		static inline bool solve(const Mat3& A, const Vec3& b, Vec3& x) {
			// 计算行列式
			FloatingType det = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
				- A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
				+ A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

			if (std::abs(det) < EPSILON) {
				x.setZero();
				return false;
			}

			FloatingType invDet = 1.0f / det;

			// 伴随矩阵 (余子式矩阵的转置)
			Mat3 adj;
			adj(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
			adj(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
			adj(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
			adj(1, 0) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
			adj(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
			adj(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
			adj(2, 0) = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
			adj(2, 1) = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
			adj(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

			x = invDet * (adj * b);
			return true;
		}

		static inline bool solveWithRegularization(const Mat3& A, const Vec3& b, Vec3& x,
			FloatingType reg = 1e-6f) {
			Mat3 A_reg = A;
			A_reg(0, 0) += reg; A_reg(1, 1) += reg; A_reg(2, 2) += reg;
			return solve(A_reg, b, x);
		}
	};

	/**
	 * @brief 性能计时器
	 */
	class PerfTimer {
	public:
		inline void start(const char* name) {
			currentName = name;
			startTime = std::chrono::high_resolution_clock::now();
		}

		inline void end() {
			auto endTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<Real, std::milli> elapsed = endTime - startTime;
			timingData[currentName] += elapsed.count();
			callCount[currentName]++;
		}

		inline void reset() {
			timingData.clear();
			callCount.clear();
		}

		inline void printReport() const {
			printf("\n========== Performance Report ==========\n");
			printf("%-40s %12s %12s %10s\n", "Function", "Total(ms)", "Avg(ms)", "Calls");
			printf("------------------------------------------------------------\n");

			Real totalTime = 0.0;
			for (auto& p : timingData) totalTime += p.second;

			for (auto& p : timingData) {
				Real avg = p.second / callCount.at(p.first);
				Real pct = (totalTime > 0) ? (p.second / totalTime * 100.0) : 0.0;
				printf("%-40s %12.2f %12.4f %10d (%5.1f%%)\n",
					p.first.c_str(), p.second, avg, callCount.at(p.first), pct);
			}
			printf("------------------------------------------------------------\n");
			printf("%-40s %12.2f ms\n", "TOTAL", totalTime);
			printf("========================================\n\n");
		}

		inline Real getTime(const char* name) const {
			auto it = timingData.find(name);
			return (it != timingData.end()) ? it->second : 0.0;
		}

	private:
		std::map<std::string, Real> timingData;
		std::map<std::string, int> callCount;
		std::chrono::high_resolution_clock::time_point startTime;
		const char* currentName = "";
	};

	// 全局计时器实例
	extern PerfTimer g_perfTimer;

} // namespace VBD
#include "GraphColoring.h"
#include <algorithm>
#include <omp.h>

namespace VBD {

    std::unordered_set<IdType> GraphColoring::getNeighbors(
        IdType vertexId,
        const std::vector<std::vector<IdType>>& vertAdjacentTets,
        const std::vector<TetVIdsArr>& tets
    ) {
        std::unordered_set<IdType> neighbors;

        // 遍历该顶点相邻的所有四面体
        for (IdType tetId : vertAdjacentTets[vertexId]) {
            const auto& tet = tets[tetId];
            // 将四面体的其他三个顶点加入邻居集合
            for (int i = 0; i < 4; ++i) {
                if (tet[i] != vertexId) {
                    neighbors.insert(tet[i]);
                }
            }
        }

        return neighbors;
    }

    std::vector<std::vector<IdType>> GraphColoring::colorVertices(
        const std::vector<std::vector<IdType>>& vertAdjacentTets,
        const std::vector<TetVIdsArr>& tets,
        size_t numVerts
    ) {
        std::vector<IdType> colors(numVerts, -1);
        std::vector<std::vector<IdType>> colorGroups;

        // 贪心着色算法
        for (size_t v = 0; v < numVerts; ++v) {
            auto neighbors = getNeighbors(static_cast<IdType>(v), vertAdjacentTets, tets);

            // 找出邻居已使用的颜色
            std::unordered_set<IdType> usedColors;
            for (IdType neighbor : neighbors) {
                if (colors[neighbor] != -1) {
                    usedColors.insert(colors[neighbor]);
                }
            }

            // 找到最小的可用颜色
            IdType newColor = 0;
            while (usedColors.count(newColor) > 0) {
                ++newColor;
            }

            colors[v] = newColor;

            // 扩展颜色组
            if (static_cast<size_t>(newColor) >= colorGroups.size()) {
                colorGroups.resize(newColor + 1);
            }
            colorGroups[newColor].push_back(static_cast<IdType>(v));
        }

        VBD_LOG("Graph coloring completed with " << colorGroups.size() << " colors");

        return colorGroups;
    }

} // namespace VBD
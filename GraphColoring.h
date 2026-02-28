#pragma once

#include "Types.h"
#include <vector>
#include <unordered_set>

namespace VBD {

    /**
     * @brief 图着色工具类
     *
     * 用于顶点着色，使得相邻顶点不在同一颜色组
     * 这样可以并行处理同一颜色的所有顶点
     */
    class GraphColoring {
    public:
        /**
         * @brief 对网格顶点进行着色
         *
         * @param vertAdjacentTets 顶点相邻的四面体列表
         * @param tets 四面体顶点索引
         * @param numVerts 顶点数量
         * @return 颜色分组，colors[c] 包含所有颜色为 c 的顶点索引
         */
        static std::vector<std::vector<IdType>> colorVertices(
            const std::vector<std::vector<IdType>>& vertAdjacentTets,
            const std::vector<TetVIdsArr>& tets,
            size_t numVerts
        );

        /**
         * @brief 获取颜色数量
         */
        static size_t getNumColors(const std::vector<std::vector<IdType>>& colors) {
            return colors.size();
        }

    private:
        /**
         * @brief 获取顶点的所有相邻顶点（通过共享四面体）
         */
        static std::unordered_set<IdType> getNeighbors(
            IdType vertexId,
            const std::vector<std::vector<IdType>>& vertAdjacentTets,
            const std::vector<TetVIdsArr>& tets
        );
    };

} // namespace VBD
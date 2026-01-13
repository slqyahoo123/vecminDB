// 评分模块
// 计算优化目标的得分函数

use crate::vector::benchmark::BenchmarkResult;
use super::config::OptimizationTarget;

/// 根据优化目标计算得分
pub fn calculate_score(result: &BenchmarkResult, target: OptimizationTarget) -> f64 {
    match target {
        OptimizationTarget::QuerySpeed => {
            // 越快越好，所以是负分
            -(result.avg_query_time_ms as f64)
        },
        OptimizationTarget::BuildSpeed => {
            // 越快越好，所以是负分
            -(result.build_time_ms as f64)
        },
        OptimizationTarget::MemoryUsage => {
            // 越小越好，所以是负分
            -(result.memory_usage_bytes as f64)
        },
        OptimizationTarget::Accuracy => {
            // 准确率越高越好
            result.accuracy
        },
        OptimizationTarget::BalancedPerformance => {
            // 综合考虑速度和准确率
            let accuracy_score = result.accuracy;
            let speed_score = -(result.avg_query_time_ms as f64) / 100.0; // 归一化
            
            // 权重平衡
            0.7 * accuracy_score + 0.3 * speed_score
        }
    }
} 
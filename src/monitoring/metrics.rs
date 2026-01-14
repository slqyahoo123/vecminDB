//! 指标定义模块
//! 
//! 定义各种监控指标类型

use serde::{Serialize, Deserialize};

/// 向量操作指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetrics {
    /// 插入操作数
    pub insert_count: u64,
    /// 删除操作数
    pub delete_count: u64,
    /// 更新操作数
    pub update_count: u64,
    /// 总向量数
    pub total_vectors: u64,
    /// 平均插入延迟（毫秒）
    pub avg_insert_latency_ms: f64,
    /// 平均删除延迟（毫秒）
    pub avg_delete_latency_ms: f64,
}

impl Default for VectorMetrics {
    fn default() -> Self {
        Self {
            insert_count: 0,
            delete_count: 0,
            update_count: 0,
            total_vectors: 0,
            avg_insert_latency_ms: 0.0,
            avg_delete_latency_ms: 0.0,
        }
    }
}

/// 索引性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetrics {
    /// 索引类型
    pub index_type: String,
    /// 索引大小（字节）
    pub index_size_bytes: usize,
    /// 构建时间（秒）
    pub build_time_secs: f64,
    /// 最后更新时间
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// 索引效率（0-1）
    pub efficiency: f64,
}

impl Default for IndexMetrics {
    fn default() -> Self {
        Self {
            index_type: "Unknown".to_string(),
            index_size_bytes: 0,
            build_time_secs: 0.0,
            last_updated: chrono::Utc::now(),
            efficiency: 1.0,
        }
    }
}

/// 查询性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// 查询总数
    pub total_queries: u64,
    /// 成功查询数
    pub successful_queries: u64,
    /// 失败查询数
    pub failed_queries: u64,
    /// 平均查询延迟（毫秒）
    pub avg_query_latency_ms: f64,
    /// P50延迟（毫秒）
    pub p50_latency_ms: f64,
    /// P95延迟（毫秒）
    pub p95_latency_ms: f64,
    /// P99延迟（毫秒）
    pub p99_latency_ms: f64,
    /// 平均召回率
    pub avg_recall: f64,
    /// QPS（每秒查询数）
    pub qps: f64,
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            avg_query_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            avg_recall: 0.0,
            qps: 0.0,
        }
    }
}

/// 资源使用指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU使用率（0-1）
    pub cpu_usage: f64,
    /// 内存使用量（字节）
    pub memory_usage_bytes: usize,
    /// 内存使用率（0-1）
    pub memory_usage_ratio: f64,
    /// GPU使用率（0-1）
    pub gpu_usage: Option<f64>,
    /// 磁盘使用量（字节）
    pub disk_usage_bytes: usize,
    /// 网络接收字节数
    pub network_rx_bytes: u64,
    /// 网络发送字节数
    pub network_tx_bytes: u64,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_bytes: 0,
            memory_usage_ratio: 0.0,
            gpu_usage: None,
            disk_usage_bytes: 0,
            network_rx_bytes: 0,
            network_tx_bytes: 0,
        }
    }
}

/// 指标快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// 快照时间
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 向量指标
    pub vector_metrics: VectorMetrics,
    /// 索引指标
    pub index_metrics: IndexMetrics,
    /// 查询指标
    pub query_metrics: QueryMetrics,
    /// 资源指标
    pub resource_metrics: ResourceMetrics,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            vector_metrics: VectorMetrics::default(),
            index_metrics: IndexMetrics::default(),
            query_metrics: QueryMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
        }
    }
}

impl MetricsSnapshot {
    /// 创建新的指标快照
    pub fn new() -> Self {
        Self::default()
    }

    /// 计算查询成功率
    pub fn query_success_rate(&self) -> f64 {
        if self.query_metrics.total_queries == 0 {
            0.0
        } else {
            self.query_metrics.successful_queries as f64 / self.query_metrics.total_queries as f64
        }
    }

    /// 计算查询失败率
    pub fn query_failure_rate(&self) -> f64 {
        1.0 - self.query_success_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_metrics_default() {
        let metrics = VectorMetrics::default();
        assert_eq!(metrics.insert_count, 0);
        assert_eq!(metrics.total_vectors, 0);
    }

    #[test]
    fn test_query_metrics_default() {
        let metrics = QueryMetrics::default();
        assert_eq!(metrics.total_queries, 0);
        assert_eq!(metrics.qps, 0.0);
    }

    #[test]
    fn test_metrics_snapshot() {
        let snapshot = MetricsSnapshot::new();
        assert_eq!(snapshot.query_success_rate(), 0.0);
        assert_eq!(snapshot.query_failure_rate(), 1.0);
    }

    #[test]
    fn test_query_success_rate() {
        let mut snapshot = MetricsSnapshot::new();
        snapshot.query_metrics.total_queries = 100;
        snapshot.query_metrics.successful_queries = 95;
        
        assert_eq!(snapshot.query_success_rate(), 0.95);
        assert_eq!(snapshot.query_failure_rate(), 0.05);
    }
}





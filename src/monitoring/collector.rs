//! 指标收集器模块

use crate::Result;
use super::metrics::*;
use parking_lot::RwLock;
use std::sync::Arc;

/// 收集器配置
#[derive(Debug, Clone)]
pub struct CollectorConfig {
    /// 采样间隔（秒）
    pub sample_interval_secs: u64,
    /// 是否启用详细指标
    pub enable_detailed_metrics: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            sample_interval_secs: 60,
            enable_detailed_metrics: true,
        }
    }
}

/// 指标收集器
pub struct MetricsCollector {
    config: CollectorConfig,
    snapshot: Arc<RwLock<MetricsSnapshot>>,
}

impl MetricsCollector {
    /// 创建新的指标收集器
    pub fn new(config: CollectorConfig) -> Result<Self> {
        Ok(Self {
            config,
            snapshot: Arc::new(RwLock::new(MetricsSnapshot::default())),
        })
    }

    /// 获取当前指标快照
    pub fn get_snapshot(&self) -> Result<MetricsSnapshot> {
        Ok(self.snapshot.read().clone())
    }

    /// 记录向量插入
    pub fn record_vector_insert(&self, latency_ms: f64) {
        let mut snapshot = self.snapshot.write();
        snapshot.vector_metrics.insert_count += 1;
        snapshot.vector_metrics.total_vectors += 1;
        // 更新平均延迟
        let count = snapshot.vector_metrics.insert_count as f64;
        snapshot.vector_metrics.avg_insert_latency_ms = 
            (snapshot.vector_metrics.avg_insert_latency_ms * (count - 1.0) + latency_ms) / count;
    }

    /// 记录查询
    pub fn record_query(&self, latency_ms: f64, success: bool) {
        let mut snapshot = self.snapshot.write();
        snapshot.query_metrics.total_queries += 1;
        if success {
            snapshot.query_metrics.successful_queries += 1;
        } else {
            snapshot.query_metrics.failed_queries += 1;
        }
        // 更新平均延迟
        let count = snapshot.query_metrics.total_queries as f64;
        snapshot.query_metrics.avg_query_latency_ms = 
            (snapshot.query_metrics.avg_query_latency_ms * (count - 1.0) + latency_ms) / count;
    }

    /// 重置指标
    pub fn reset(&self) -> Result<()> {
        *self.snapshot.write() = MetricsSnapshot::default();
        Ok(())
    }
}







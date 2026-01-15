//! 指标报告器模块

use crate::Result;
use super::metrics::MetricsSnapshot;

/// 报告格式
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Json,
    Text,
}

/// 指标报告器
pub struct MetricsReporter;

impl MetricsReporter {
    /// 创建新的报告器
    pub fn new() -> Self {
        Self
    }

    /// 生成报告
    pub fn generate_report(&self, snapshot: &MetricsSnapshot, format: ReportFormat) -> Result<String> {
        match format {
            ReportFormat::Json => self.generate_json_report(snapshot),
            ReportFormat::Text => self.generate_text_report(snapshot),
        }
    }

    fn generate_json_report(&self, snapshot: &MetricsSnapshot) -> Result<String> {
        serde_json::to_string_pretty(snapshot)
            .map_err(|e| crate::Error::serialization(format!("JSON序列化失败: {}", e)))
    }

    fn generate_text_report(&self, snapshot: &MetricsSnapshot) -> Result<String> {
        Ok(format!(
            "=== 向量数据库监控报告 ===\n\
            时间: {}\n\
            \n\
            向量操作:\n\
            - 插入: {}\n\
            - 删除: {}\n\
            - 总数: {}\n\
            \n\
            查询性能:\n\
            - 总查询数: {}\n\
            - 成功率: {:.2}%\n\
            - 平均延迟: {:.2}ms\n\
            \n\
            资源使用:\n\
            - CPU: {:.2}%\n\
            - 内存: {:.2}MB\n",
            snapshot.timestamp,
            snapshot.vector_metrics.insert_count,
            snapshot.vector_metrics.delete_count,
            snapshot.vector_metrics.total_vectors,
            snapshot.query_metrics.total_queries,
            snapshot.query_success_rate() * 100.0,
            snapshot.query_metrics.avg_query_latency_ms,
            snapshot.resource_metrics.cpu_usage * 100.0,
            snapshot.resource_metrics.memory_usage_bytes as f64 / 1024.0 / 1024.0,
        ))
    }
}






//! 性能分析器模块

use crate::Result;
use super::{MetricsCollector, metrics::MetricsSnapshot};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// 性能问题
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIssue {
    /// 问题类型
    pub issue_type: String,
    /// 严重程度（1-10）
    pub severity: u8,
    /// 描述
    pub description: String,
    /// 建议
    pub recommendation: String,
}

/// 性能报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// 报告时间
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 整体评分（0-100）
    pub overall_score: f64,
    /// 发现的问题
    pub issues: Vec<PerformanceIssue>,
    /// 指标快照
    pub snapshot: MetricsSnapshot,
}

/// 性能分析器
pub struct PerformanceAnalyzer {
    collector: Arc<MetricsCollector>,
}

impl PerformanceAnalyzer {
    /// 创建新的性能分析器
    pub fn new(collector: Arc<MetricsCollector>) -> Self {
        Self { collector }
    }

    /// 分析性能
    pub fn analyze(&self) -> Result<PerformanceReport> {
        let snapshot = self.collector.get_snapshot()?;
        let mut issues = Vec::new();
        let mut score = 100.0;

        // 分析查询性能
        if snapshot.query_metrics.avg_query_latency_ms > 100.0 {
            issues.push(PerformanceIssue {
                issue_type: "HighQueryLatency".to_string(),
                severity: 7,
                description: format!("平均查询延迟过高: {:.2}ms", snapshot.query_metrics.avg_query_latency_ms),
                recommendation: "考虑优化索引参数或增加资源".to_string(),
            });
            score -= 20.0;
        }

        // 分析资源使用
        if snapshot.resource_metrics.cpu_usage > 0.9 {
            issues.push(PerformanceIssue {
                issue_type: "HighCPUUsage".to_string(),
                severity: 8,
                description: format!("CPU使用率过高: {:.2}%", snapshot.resource_metrics.cpu_usage * 100.0),
                recommendation: "考虑增加CPU资源或优化并发策略".to_string(),
            });
            score -= 15.0;
        }

        Ok(PerformanceReport {
            timestamp: chrono::Utc::now(),
            overall_score: score.max(0.0),
            issues,
            snapshot,
        })
    }
}







//! 监控系统模块
//! 
//! 提供向量数据库操作的全面监控功能。
//! 
//! ## 功能特性
//! 
//! - 向量操作指标收集
//! - 索引性能监控
//! - 查询延迟统计
//! - 资源使用监控
//! - 实时性能分析

pub mod metrics;
pub mod collector;
pub mod reporter;
pub mod analyzer;

// 重新导出核心类型
pub use metrics::{
    VectorMetrics,
    IndexMetrics,
    QueryMetrics,
    ResourceMetrics,
    MetricsSnapshot,
};

pub use collector::{
    MetricsCollector,
    CollectorConfig,
};

pub use reporter::{
    MetricsReporter,
    ReportFormat,
};

pub use analyzer::{
    PerformanceAnalyzer,
    PerformanceReport,
    PerformanceIssue,
};

use crate::Result;
use std::sync::Arc;
use parking_lot::RwLock;

/// 监控系统
pub struct MonitoringSystem {
    /// 指标收集器
    collector: Arc<MetricsCollector>,
    /// 指标报告器
    reporter: Arc<MetricsReporter>,
    /// 性能分析器
    analyzer: Arc<PerformanceAnalyzer>,
    /// 是否启用
    enabled: Arc<RwLock<bool>>,
}

impl MonitoringSystem {
    /// 创建新的监控系统
    pub fn new(config: CollectorConfig) -> Result<Self> {
        let collector = Arc::new(MetricsCollector::new(config)?);
        let reporter = Arc::new(MetricsReporter::new());
        let analyzer = Arc::new(PerformanceAnalyzer::new(collector.clone()));

        Ok(Self {
            collector,
            reporter,
            analyzer,
            enabled: Arc::new(RwLock::new(true)),
        })
    }

    /// 创建默认监控系统
    pub fn default() -> Result<Self> {
        Self::new(CollectorConfig::default())
    }

    /// 启用监控
    pub fn enable(&self) {
        *self.enabled.write() = true;
    }

    /// 禁用监控
    pub fn disable(&self) {
        *self.enabled.write() = false;
    }

    /// 检查是否启用
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// 获取指标收集器
    pub fn collector(&self) -> Arc<MetricsCollector> {
        self.collector.clone()
    }

    /// 获取指标报告器
    pub fn reporter(&self) -> Arc<MetricsReporter> {
        self.reporter.clone()
    }

    /// 获取性能分析器
    pub fn analyzer(&self) -> Arc<PerformanceAnalyzer> {
        self.analyzer.clone()
    }

    /// 获取当前指标快照
    pub fn get_snapshot(&self) -> Result<MetricsSnapshot> {
        self.collector.get_snapshot()
    }

    /// 生成性能报告
    pub fn generate_report(&self) -> Result<PerformanceReport> {
        self.analyzer.analyze()
    }

    /// 重置所有指标
    pub fn reset(&self) -> Result<()> {
        self.collector.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_system_creation() {
        let system = MonitoringSystem::default();
        assert!(system.is_ok());
    }

    #[test]
    fn test_enable_disable() {
        let system = MonitoringSystem::default().unwrap();
        assert!(system.is_enabled());

        system.disable();
        assert!(!system.is_enabled());

        system.enable();
        assert!(system.is_enabled());
    }
}





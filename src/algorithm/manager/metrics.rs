use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};
use log::{info, debug};

use crate::{Error, Result};

// 重新导出executor模块中的ExecutionMetrics
pub use crate::algorithm::executor::metrics::ExecutionMetrics;

/// 算法执行指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    /// 算法ID
    pub algorithm_id: String,
    /// 执行ID
    pub execution_id: String,
    /// 开始时间
    pub start_time: SystemTime,
    /// 结束时间
    pub end_time: Option<SystemTime>,
    /// 执行时间(毫秒)
    pub execution_time_ms: u64,
    /// 内存使用峰值(字节)
    pub peak_memory_bytes: usize,
    /// CPU使用率(0-100)
    pub cpu_usage_percent: f64,
    /// 输入数据大小(字节)
    pub input_size_bytes: usize,
    /// 输出数据大小(字节)
    pub output_size_bytes: usize,
    /// 错误信息
    pub error_message: Option<String>,
    /// 自定义指标
    pub custom_metrics: HashMap<String, f64>,
    /// 性能等级
    pub performance_grade: PerformanceGrade,
}

/// 性能等级
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PerformanceGrade {
    /// 优秀
    Excellent,
    /// 良好
    Good,
    /// 一般
    Average,
    /// 差
    Poor,
    /// 超时
    Timeout,
}

/// 算法性能统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmStatistics {
    /// 算法ID
    pub algorithm_id: String,
    /// 总执行次数
    pub total_executions: u64,
    /// 成功执行次数
    pub successful_executions: u64,
    /// 失败执行次数
    pub failed_executions: u64,
    /// 平均执行时间(毫秒)
    pub average_execution_time_ms: f64,
    /// 最短执行时间(毫秒)
    pub min_execution_time_ms: u64,
    /// 最长执行时间(毫秒)
    pub max_execution_time_ms: u64,
    /// 平均内存使用(字节)
    pub average_memory_bytes: f64,
    /// 内存使用峰值(字节)
    pub peak_memory_bytes: usize,
    /// 平均CPU使用率
    pub average_cpu_usage: f64,
    /// 成功率
    pub success_rate: f64,
    /// 性能趋势
    pub performance_trend: Vec<f64>,
    /// 最后更新时间
    pub last_updated: SystemTime,
}

/// 指标收集器
pub struct MetricsCollector {
    /// 当前执行指标
    current_metrics: Arc<RwLock<HashMap<String, AlgorithmMetrics>>>,
    /// 历史统计
    statistics: Arc<RwLock<HashMap<String, AlgorithmStatistics>>>,
    /// 配置
    config: MetricsConfig,
}

/// 指标配置
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// 是否启用性能监控
    pub enable_performance_monitoring: bool,
    /// 历史记录保留天数
    pub retention_days: u32,
    /// 统计更新间隔
    pub update_interval_ms: u64,
    /// 性能阈值配置
    pub performance_thresholds: PerformanceThresholds,
}

/// 性能阈值配置
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// 优秀执行时间阈值(毫秒)
    pub excellent_execution_time_ms: u64,
    /// 良好执行时间阈值(毫秒)
    pub good_execution_time_ms: u64,
    /// 一般执行时间阈值(毫秒)
    pub average_execution_time_ms: u64,
    /// 内存使用阈值(MB)
    pub memory_threshold_mb: usize,
    /// CPU使用率阈值
    pub cpu_threshold_percent: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            retention_days: 30,
            update_interval_ms: 1000,
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            excellent_execution_time_ms: 100,
            good_execution_time_ms: 500,
            average_execution_time_ms: 2000,
            memory_threshold_mb: 100,
            cpu_threshold_percent: 80.0,
        }
    }
}

impl MetricsCollector {
    /// 创建新的指标收集器
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            current_metrics: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// 开始收集指标
    pub fn start_collection(&self, algorithm_id: &str, execution_id: &str) -> Result<()> {
        if !self.config.enable_performance_monitoring {
            return Ok(());
        }

        let metrics = AlgorithmMetrics {
            algorithm_id: algorithm_id.to_string(),
            execution_id: execution_id.to_string(),
            start_time: SystemTime::now(),
            end_time: None,
            execution_time_ms: 0,
            peak_memory_bytes: 0,
            cpu_usage_percent: 0.0,
            input_size_bytes: 0,
            output_size_bytes: 0,
            error_message: None,
            custom_metrics: HashMap::new(),
            performance_grade: PerformanceGrade::Average,
        };

        let mut current = self.current_metrics.write().map_err(|e| {
            Error::lock(format!("无法获取当前指标锁: {}", e))
        })?;

        current.insert(execution_id.to_string(), metrics);
        debug!("开始收集算法指标: algorithm_id={}, execution_id={}", algorithm_id, execution_id);

        Ok(())
    }

    /// 结束指标收集
    pub fn end_collection(&self, execution_id: &str, success: bool, error_message: Option<String>) -> Result<AlgorithmMetrics> {
        if !self.config.enable_performance_monitoring {
            return Err(Error::invalid_state("性能监控未启用"));
        }

        let mut current = self.current_metrics.write().map_err(|e| {
            Error::lock(format!("无法获取当前指标锁: {}", e))
        })?;

        if let Some(mut metrics) = current.remove(execution_id) {
            metrics.end_time = Some(SystemTime::now());
            metrics.execution_time_ms = metrics.end_time.unwrap()
                .duration_since(metrics.start_time)
                .unwrap_or_default()
                .as_millis() as u64;

            metrics.error_message = error_message;
            metrics.performance_grade = self.calculate_performance_grade(&metrics);

            // 更新统计信息
            self.update_statistics(&metrics, success)?;

            debug!("结束指标收集: execution_id={}, execution_time={}ms", execution_id, metrics.execution_time_ms);
            Ok(metrics)
        } else {
            Err(Error::not_found(format!("未找到执行ID的指标: {}", execution_id)))
        }
    }

    /// 更新内存使用
    pub fn update_memory_usage(&self, execution_id: &str, memory_bytes: usize) -> Result<()> {
        if !self.config.enable_performance_monitoring {
            return Ok(());
        }

        let mut current = self.current_metrics.write().map_err(|e| {
            Error::lock(format!("无法获取当前指标锁: {}", e))
        })?;

        if let Some(metrics) = current.get_mut(execution_id) {
            if memory_bytes > metrics.peak_memory_bytes {
                metrics.peak_memory_bytes = memory_bytes;
            }
        }

        Ok(())
    }

    /// 更新CPU使用率
    pub fn update_cpu_usage(&self, execution_id: &str, cpu_percent: f64) -> Result<()> {
        if !self.config.enable_performance_monitoring {
            return Ok(());
        }

        let mut current = self.current_metrics.write().map_err(|e| {
            Error::lock(format!("无法获取当前指标锁: {}", e))
        })?;

        if let Some(metrics) = current.get_mut(execution_id) {
            if cpu_percent > metrics.cpu_usage_percent {
                metrics.cpu_usage_percent = cpu_percent;
            }
        }

        Ok(())
    }

    /// 设置输入输出大小
    pub fn set_data_sizes(&self, execution_id: &str, input_size: usize, output_size: usize) -> Result<()> {
        if !self.config.enable_performance_monitoring {
            return Ok(());
        }

        let mut current = self.current_metrics.write().map_err(|e| {
            Error::lock(format!("无法获取当前指标锁: {}", e))
        })?;

        if let Some(metrics) = current.get_mut(execution_id) {
            metrics.input_size_bytes = input_size;
            metrics.output_size_bytes = output_size;
        }

        Ok(())
    }

    /// 添加自定义指标
    pub fn add_custom_metric(&self, execution_id: &str, name: &str, value: f64) -> Result<()> {
        if !self.config.enable_performance_monitoring {
            return Ok(());
        }

        let mut current = self.current_metrics.write().map_err(|e| {
            Error::lock(format!("无法获取当前指标锁: {}", e))
        })?;

        if let Some(metrics) = current.get_mut(execution_id) {
            metrics.custom_metrics.insert(name.to_string(), value);
        }

        Ok(())
    }

    /// 获取算法统计信息
    pub fn get_algorithm_statistics(&self, algorithm_id: &str) -> Result<Option<AlgorithmStatistics>> {
        let statistics = self.statistics.read().map_err(|e| {
            Error::lock(format!("无法获取统计锁: {}", e))
        })?;

        Ok(statistics.get(algorithm_id).cloned())
    }

    /// 获取所有统计信息
    pub fn get_all_statistics(&self) -> Result<HashMap<String, AlgorithmStatistics>> {
        let statistics = self.statistics.read().map_err(|e| {
            Error::lock(format!("无法获取统计锁: {}", e))
        })?;

        Ok(statistics.clone())
    }

    /// 清理过期指标
    pub fn cleanup_expired_metrics(&self) -> Result<usize> {
        let retention_duration = Duration::from_secs(self.config.retention_days as u64 * 24 * 60 * 60);
        let cutoff_time = SystemTime::now() - retention_duration;

        let mut statistics = self.statistics.write().map_err(|e| {
            Error::lock(format!("无法获取统计锁: {}", e))
        })?;

        let initial_count = statistics.len();
        statistics.retain(|_, stat| stat.last_updated >= cutoff_time);
        let cleaned_count = initial_count - statistics.len();

        if cleaned_count > 0 {
            info!("清理了 {} 个过期的算法统计记录", cleaned_count);
        }

        Ok(cleaned_count)
    }

    /// 计算性能等级
    fn calculate_performance_grade(&self, metrics: &AlgorithmMetrics) -> PerformanceGrade {
        let thresholds = &self.config.performance_thresholds;

        // 如果有错误，直接返回差
        if metrics.error_message.is_some() {
            return PerformanceGrade::Poor;
        }

        // 根据执行时间判断
        if metrics.execution_time_ms <= thresholds.excellent_execution_time_ms {
            PerformanceGrade::Excellent
        } else if metrics.execution_time_ms <= thresholds.good_execution_time_ms {
            PerformanceGrade::Good
        } else if metrics.execution_time_ms <= thresholds.average_execution_time_ms {
            PerformanceGrade::Average
        } else {
            PerformanceGrade::Poor
        }
    }

    /// 更新统计信息
    fn update_statistics(&self, metrics: &AlgorithmMetrics, success: bool) -> Result<()> {
        let mut statistics = self.statistics.write().map_err(|e| {
            Error::lock(format!("无法获取统计锁: {}", e))
        })?;

        let stat = statistics.entry(metrics.algorithm_id.clone())
            .or_insert_with(|| AlgorithmStatistics {
                algorithm_id: metrics.algorithm_id.clone(),
                total_executions: 0,
                successful_executions: 0,
                failed_executions: 0,
                average_execution_time_ms: 0.0,
                min_execution_time_ms: u64::MAX,
                max_execution_time_ms: 0,
                average_memory_bytes: 0.0,
                peak_memory_bytes: 0,
                average_cpu_usage: 0.0,
                success_rate: 0.0,
                performance_trend: Vec::new(),
                last_updated: SystemTime::now(),
            });

        // 更新基本计数
        stat.total_executions += 1;
        if success {
            stat.successful_executions += 1;
        } else {
            stat.failed_executions += 1;
        }

        // 更新执行时间统计
        let exec_time = metrics.execution_time_ms;
        stat.average_execution_time_ms = (stat.average_execution_time_ms * (stat.total_executions - 1) as f64 + exec_time as f64) / stat.total_executions as f64;
        stat.min_execution_time_ms = stat.min_execution_time_ms.min(exec_time);
        stat.max_execution_time_ms = stat.max_execution_time_ms.max(exec_time);

        // 更新内存统计
        stat.average_memory_bytes = (stat.average_memory_bytes * (stat.total_executions - 1) as f64 + metrics.peak_memory_bytes as f64) / stat.total_executions as f64;
        stat.peak_memory_bytes = stat.peak_memory_bytes.max(metrics.peak_memory_bytes);

        // 更新CPU统计
        stat.average_cpu_usage = (stat.average_cpu_usage * (stat.total_executions - 1) as f64 + metrics.cpu_usage_percent) / stat.total_executions as f64;

        // 更新成功率
        stat.success_rate = stat.successful_executions as f64 / stat.total_executions as f64;

        // 更新性能趋势
        stat.performance_trend.push(exec_time as f64);
        if stat.performance_trend.len() > 100 {
            stat.performance_trend.remove(0);
        }

        stat.last_updated = SystemTime::now();

        Ok(())
    }
}

/// 指标报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    /// 报告生成时间
    pub generated_at: SystemTime,
    /// 总算法数量
    pub total_algorithms: usize,
    /// 总执行次数
    pub total_executions: u64,
    /// 总成功次数
    pub total_successful: u64,
    /// 总失败次数
    pub total_failed: u64,
    /// 整体成功率
    pub overall_success_rate: f64,
    /// 平均执行时间
    pub average_execution_time_ms: f64,
    /// 性能分布
    pub performance_distribution: HashMap<PerformanceGrade, u64>,
    /// 最佳性能算法
    pub best_performing_algorithms: Vec<String>,
    /// 最差性能算法
    pub worst_performing_algorithms: Vec<String>,
}

/// 生成指标报告
pub fn generate_metrics_report(collector: &MetricsCollector) -> Result<MetricsReport> {
    let all_stats = collector.get_all_statistics()?;
    
    if all_stats.is_empty() {
        return Ok(MetricsReport {
            generated_at: SystemTime::now(),
            total_algorithms: 0,
            total_executions: 0,
            total_successful: 0,
            total_failed: 0,
            overall_success_rate: 0.0,
            average_execution_time_ms: 0.0,
            performance_distribution: HashMap::new(),
            best_performing_algorithms: Vec::new(),
            worst_performing_algorithms: Vec::new(),
        });
    }

    let total_algorithms = all_stats.len();
    let total_executions = all_stats.values().map(|s| s.total_executions).sum();
    let total_successful = all_stats.values().map(|s| s.successful_executions).sum();
    let total_failed = all_stats.values().map(|s| s.failed_executions).sum();
    let overall_success_rate = if total_executions > 0 {
        total_successful as f64 / total_executions as f64
    } else {
        0.0
    };

    let average_execution_time_ms = if total_executions > 0 {
        all_stats.values()
            .map(|s| s.average_execution_time_ms * s.total_executions as f64)
            .sum::<f64>() / total_executions as f64
    } else {
        0.0
    };

    // 计算性能分布（简化版本）
    let mut performance_distribution = HashMap::new();
    for stat in all_stats.values() {
        let grade = if stat.average_execution_time_ms <= 100.0 {
            PerformanceGrade::Excellent
        } else if stat.average_execution_time_ms <= 500.0 {
            PerformanceGrade::Good
        } else if stat.average_execution_time_ms <= 2000.0 {
            PerformanceGrade::Average
        } else {
            PerformanceGrade::Poor
        };
        *performance_distribution.entry(grade).or_insert(0) += stat.total_executions;
    }

    // 找出最佳和最差性能算法
    let mut sorted_by_performance: Vec<_> = all_stats.iter().collect();
    sorted_by_performance.sort_by(|a, b| a.1.average_execution_time_ms.partial_cmp(&b.1.average_execution_time_ms).unwrap_or(std::cmp::Ordering::Equal));

    let best_performing_algorithms = sorted_by_performance.iter()
        .take(5)
        .map(|(id, _)| id.to_string())
        .collect();

    let worst_performing_algorithms = sorted_by_performance.iter()
        .rev()
        .take(5)
        .map(|(id, _)| id.to_string())
        .collect();

    Ok(MetricsReport {
        generated_at: SystemTime::now(),
        total_algorithms,
        total_executions,
        total_successful,
        total_failed,
        overall_success_rate,
        average_execution_time_ms,
        performance_distribution,
        best_performing_algorithms,
        worst_performing_algorithms,
    })
} 
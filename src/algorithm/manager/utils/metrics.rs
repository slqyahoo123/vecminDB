// 指标工具

use std::time::{Duration, SystemTime};
use std::collections::HashMap;

/// 执行指标
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time: Duration,
    pub peak_memory_usage: u64,
    pub total_cpu_time: Duration,
    pub last_execution_time: Option<SystemTime>,
}

impl ExecutionMetrics {
    /// 更新执行指标
    pub fn update_execution(&mut self, success: bool, execution_time: Duration, memory_usage: u64) {
        self.total_executions += 1;
        
        if success {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }
        
        // 更新平均执行时间
        let total_time = self.average_execution_time * (self.total_executions - 1) as u32 + execution_time;
        self.average_execution_time = total_time / self.total_executions as u32;
        
        // 更新峰值内存使用
        if memory_usage > self.peak_memory_usage {
            self.peak_memory_usage = memory_usage;
        }
        
        self.last_execution_time = Some(SystemTime::now());
    }
    
    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }
    
    /// 获取失败率
    pub fn failure_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.failed_executions as f64 / self.total_executions as f64
        }
    }
}

/// 算法指标
#[derive(Debug, Clone, Default)]
pub struct AlgorithmMetrics {
    pub algorithm_id: String,
    pub execution_metrics: ExecutionMetrics,
    pub custom_metrics: HashMap<String, f64>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

impl AlgorithmMetrics {
    /// 创建新的算法指标
    pub fn new(algorithm_id: String) -> Self {
        let now = SystemTime::now();
        Self {
            algorithm_id,
            execution_metrics: ExecutionMetrics::default(),
            custom_metrics: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
    
    /// 更新自定义指标
    pub fn update_custom_metric(&mut self, key: String, value: f64) {
        self.custom_metrics.insert(key, value);
        self.updated_at = SystemTime::now();
    }
    
    /// 获取自定义指标
    pub fn get_custom_metric(&self, key: &str) -> Option<f64> {
        self.custom_metrics.get(key).copied()
    }
} 
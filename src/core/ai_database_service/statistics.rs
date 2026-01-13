/// AI数据库服务统计模块
/// 
/// 负责收集和管理服务的统计信息

use serde::{Serialize, Deserialize};

/// AI数据库服务统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIDatabaseStatistics {
    /// 总请求数
    pub total_requests: u64,
    /// 成功请求数
    pub successful_requests: u64,
    /// 失败请求数
    pub failed_requests: u64,
    /// 数据转换次数
    pub data_conversions: u64,
    /// 训练任务数
    pub training_tasks: u64,
    /// 算法定制次数
    pub algorithm_customizations: u64,
    /// 平均响应时间（毫秒）
    pub average_response_time_ms: f64,
    /// 存储的模型数量
    pub stored_models_count: usize,
    /// 存储的数据集数量
    pub stored_datasets_count: usize,
}

impl AIDatabaseStatistics {
    /// 创建新的统计实例
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            data_conversions: 0,
            training_tasks: 0,
            algorithm_customizations: 0,
            average_response_time_ms: 0.0,
            stored_models_count: 0,
            stored_datasets_count: 0,
        }
    }

    /// 增加请求计数
    pub fn increment_total_requests(&mut self) {
        self.total_requests += 1;
    }

    /// 增加成功请求计数
    pub fn increment_successful_requests(&mut self) {
        self.successful_requests += 1;
    }

    /// 增加失败请求计数
    pub fn increment_failed_requests(&mut self) {
        self.failed_requests += 1;
    }

    /// 更新平均响应时间
    pub fn update_average_response_time(&mut self, new_response_time: f64) {
        if self.successful_requests > 0 {
            let total_requests = self.successful_requests as f64;
            self.average_response_time_ms = 
                (self.average_response_time_ms * (total_requests - 1.0) + new_response_time) / total_requests;
        } else {
            self.average_response_time_ms = new_response_time;
        }
    }

    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }

    /// 获取失败率
    pub fn failure_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.failed_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

impl Default for AIDatabaseStatistics {
    fn default() -> Self {
        Self::new()
    }
} 
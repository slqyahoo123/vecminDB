/// 健康检查模块
/// 
/// 负责服务健康状态的监控和报告

use serde::{Serialize, Deserialize};

/// 健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// 整体健康状态
    pub overall_health: OverallHealth,
    /// 组件状态列表
    pub components_status: Vec<(String, bool)>,
    /// 运行时间（秒）
    pub uptime_seconds: u64,
    /// 最后检查时间
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// 整体健康状态枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverallHealth {
    /// 健康
    Healthy,
    /// 降级
    Degraded,
    /// 不健康
    Unhealthy,
}

impl HealthStatus {
    /// 创建新的健康状态
    pub fn new() -> Self {
        Self {
            overall_health: OverallHealth::Healthy,
            components_status: Vec::new(),
            uptime_seconds: 0,
            last_check: chrono::Utc::now(),
        }
    }

    /// 添加组件状态
    pub fn add_component_status(&mut self, component_name: String, is_healthy: bool) {
        self.components_status.push((component_name, is_healthy));
    }

    /// 更新整体健康状态
    pub fn update_overall_health(&mut self) {
        let healthy_count = self.components_status.iter().filter(|(_, healthy)| *healthy).count();
        let total_count = self.components_status.len();
        
        if total_count == 0 {
            self.overall_health = OverallHealth::Healthy;
        } else if healthy_count == total_count {
            self.overall_health = OverallHealth::Healthy;
        } else if healthy_count > 0 {
            self.overall_health = OverallHealth::Degraded;
        } else {
            self.overall_health = OverallHealth::Unhealthy;
        }
    }

    /// 设置运行时间
    pub fn set_uptime(&mut self, uptime_seconds: u64) {
        self.uptime_seconds = uptime_seconds;
    }

    /// 更新检查时间
    pub fn update_check_time(&mut self) {
        self.last_check = chrono::Utc::now();
    }

    /// 获取健康的组件数量
    pub fn healthy_components_count(&self) -> usize {
        self.components_status.iter().filter(|(_, healthy)| *healthy).count()
    }

    /// 获取不健康的组件数量
    pub fn unhealthy_components_count(&self) -> usize {
        self.components_status.iter().filter(|(_, healthy)| !*healthy).count()
    }

    /// 检查是否所有组件都健康
    pub fn is_all_components_healthy(&self) -> bool {
        self.components_status.iter().all(|(_, healthy)| *healthy)
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::new()
    }
} 
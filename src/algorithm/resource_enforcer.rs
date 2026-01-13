use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use log::{warn, error, debug, info};

use crate::error::{Result};

/// 资源类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU 使用率 (0-100%)
    CPU,
    /// 内存使用量 (字节)
    Memory,
    /// 磁盘读取量 (字节)
    DiskRead,
    /// 磁盘写入量 (字节)
    DiskWrite,
    /// 网络接收量 (字节)
    NetworkReceive,
    /// 网络发送量 (字节)
    NetworkSend,
    /// GPU 使用率 (0-100%)
    GPU,
    /// GPU 内存使用量 (字节)
    GPUMemory,
    /// 文件描述符数量
    FileDescriptors,
    /// 线程数量
    ThreadCount,
}

/// 资源限制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit {
    /// 资源类型
    pub resource_type: ResourceType,
    /// 软限制
    pub soft_limit: u64,
    /// 硬限制
    pub hard_limit: u64,
    /// 监控间隔
    pub check_interval: Duration,
    /// 是否启用
    pub enabled: bool,
}

/// 资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// 资源类型
    pub resource_type: ResourceType,
    /// 当前使用量
    pub current_usage: u64,
    /// 峰值使用量
    pub peak_usage: u64,
    /// 平均使用量
    pub average_usage: f64,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
    /// 采样计数
    pub sample_count: u64,
}

/// 资源监控事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringEvent {
    /// 事件ID
    pub event_id: Uuid,
    /// 资源类型
    pub resource_type: ResourceType,
    /// 事件类型
    pub event_type: ResourceEventType,
    /// 当前使用量
    pub current_usage: u64,
    /// 限制值
    pub limit_value: u64,
    /// 事件时间
    pub timestamp: DateTime<Utc>,
    /// 相关任务ID
    pub task_id: Option<Uuid>,
    /// 事件消息
    pub message: String,
    /// 严重程度
    pub severity: EventSeverity,
}

/// 资源事件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceEventType {
    /// 软限制警告
    SoftLimitWarning,
    /// 硬限制违规
    HardLimitViolation,
    /// 资源耗尽
    ResourceExhaustion,
    /// 异常峰值
    AbnormalSpike,
    /// 资源恢复正常
    ResourceRecovered,
}

/// 事件严重程度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 强制执行统计信息
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnforcementStatistics {
    /// 总检查次数
    pub total_checks: u64,
    /// 软限制违规次数
    pub soft_limit_violations: u64,
    /// 硬限制违规次数
    pub hard_limit_violations: u64,
    /// 强制终止次数
    pub forced_terminations: u64,
    /// 警告次数
    pub warnings_issued: u64,
    /// 最后检查时间
    pub last_check_time: DateTime<Utc>,
    /// 平均检查间隔
    pub average_check_interval: f64,
}

/// 资源强制执行器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEnforcerConfig {
    /// 全局启用状态
    pub enabled: bool,
    /// 监控间隔
    pub monitoring_interval: Duration,
    /// 资源限制映射
    pub resource_limits: HashMap<ResourceType, ResourceLimit>,
    /// 是否启用自动终止
    pub auto_terminate_on_violation: bool,
    /// 警告阈值 (0.0 - 1.0, 相对于限制的百分比)
    pub warning_threshold: f64,
    /// 历史记录保留时间
    pub history_retention: Duration,
    /// 最大事件数
    pub max_events: usize,
}

impl Default for ResourceEnforcerConfig {
    fn default() -> Self {
        let mut resource_limits = HashMap::new();
        
        // 默认资源限制
        resource_limits.insert(ResourceType::Memory, ResourceLimit {
            resource_type: ResourceType::Memory,
            soft_limit: 1024 * 1024 * 1024, // 1GB
            hard_limit: 2048 * 1024 * 1024, // 2GB
            check_interval: Duration::from_secs(5),
            enabled: true,
        });
        
        resource_limits.insert(ResourceType::CPU, ResourceLimit {
            resource_type: ResourceType::CPU,
            soft_limit: 80, // 80%
            hard_limit: 95, // 95%
            check_interval: Duration::from_secs(3),
            enabled: true,
        });
        
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(1),
            resource_limits,
            auto_terminate_on_violation: false,
            warning_threshold: 0.8,
            history_retention: Duration::from_hours(24),
            max_events: 10000,
        }
    }
}

/// 资源强制执行器
#[derive(Debug)]
pub struct ResourceEnforcer {
    /// 配置
    config: Arc<RwLock<ResourceEnforcerConfig>>,
    /// 当前资源使用情况
    current_usage: Arc<RwLock<HashMap<ResourceType, ResourceUsage>>>,
    /// 事件历史
    event_history: Arc<RwLock<Vec<ResourceMonitoringEvent>>>,
    /// 统计信息
    statistics: Arc<RwLock<EnforcementStatistics>>,
    /// 监控线程控制
    monitoring_active: Arc<Mutex<bool>>,
    /// 注册的任务
    registered_tasks: Arc<RwLock<HashMap<Uuid, String>>>,
}

impl ResourceEnforcer {
    /// 创建新的资源强制执行器
    pub fn new(config: ResourceEnforcerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            current_usage: Arc::new(RwLock::new(HashMap::new())),
            event_history: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(EnforcementStatistics::default())),
            monitoring_active: Arc::new(Mutex::new(false)),
            registered_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 启动资源监控
    pub fn start_monitoring(&self) -> Result<()> {
        let mut active = self.monitoring_active.lock().unwrap();
        if *active {
            return Ok(()); // 已经在监控中
        }
        *active = true;
        
        debug!("启动资源监控");
        
        // 在这里可以启动后台监控线程
        // 实际实现中需要根据具体平台获取资源使用情况
        
        Ok(())
    }
    
    /// 停止资源监控
    pub fn stop_monitoring(&self) -> Result<()> {
        let mut active = self.monitoring_active.lock().unwrap();
        *active = false;
        
        debug!("停止资源监控");
        Ok(())
    }
    
    /// 注册任务
    pub fn register_task(&self, task_id: Uuid, task_name: String) -> Result<()> {
        let mut tasks = self.registered_tasks.write().unwrap();
        tasks.insert(task_id, task_name);
        debug!("注册任务: {} ({})", task_id, tasks[&task_id]);
        Ok(())
    }
    
    /// 注销任务
    pub fn unregister_task(&self, task_id: &Uuid) -> Result<()> {
        let mut tasks = self.registered_tasks.write().unwrap();
        if let Some(task_name) = tasks.remove(task_id) {
            debug!("注销任务: {} ({})", task_id, task_name);
        }
        Ok(())
    }
    
    /// 检查资源使用情况
    pub fn check_resource_usage(&self, task_id: Option<Uuid>) -> Result<bool> {
        let config = self.config.read().unwrap();
        if !config.enabled {
            return Ok(true);
        }
        
        let mut stats = self.statistics.write().unwrap();
        stats.total_checks += 1;
        stats.last_check_time = Utc::now();
        
        let mut violation_detected = false;
        
        // 检查每种资源类型
        for (resource_type, limit) in &config.resource_limits {
            if !limit.enabled {
                continue;
            }
            
            // 获取当前使用情况（这里是模拟实现）
            let current_usage = self.get_current_resource_usage(resource_type)?;
            
            // 更新使用情况记录
            self.update_usage_record(resource_type.clone(), current_usage)?;
            
            // 检查软限制
            if current_usage > limit.soft_limit {
                stats.soft_limit_violations += 1;
                
                let event = ResourceMonitoringEvent {
                    event_id: Uuid::new_v4(),
                    resource_type: resource_type.clone(),
                    event_type: ResourceEventType::SoftLimitWarning,
                    current_usage,
                    limit_value: limit.soft_limit,
                    timestamp: Utc::now(),
                    task_id,
                    message: format!("软限制警告: {} 使用量 {} 超过限制 {}", 
                                   self.resource_type_name(resource_type), current_usage, limit.soft_limit),
                    severity: EventSeverity::Medium,
                };
                
                self.add_event(event)?;
                
                warn!("资源软限制警告: {:?} 使用量 {} 超过限制 {}", 
                      resource_type, current_usage, limit.soft_limit);
            }
            
            // 检查硬限制
            if current_usage > limit.hard_limit {
                stats.hard_limit_violations += 1;
                violation_detected = true;
                
                let event = ResourceMonitoringEvent {
                    event_id: Uuid::new_v4(),
                    resource_type: resource_type.clone(),
                    event_type: ResourceEventType::HardLimitViolation,
                    current_usage,
                    limit_value: limit.hard_limit,
                    timestamp: Utc::now(),
                    task_id,
                    message: format!("硬限制违规: {} 使用量 {} 超过限制 {}", 
                                   self.resource_type_name(resource_type), current_usage, limit.hard_limit),
                    severity: EventSeverity::High,
                };
                
                self.add_event(event)?;
                
                error!("资源硬限制违规: {:?} 使用量 {} 超过限制 {}", 
                       resource_type, current_usage, limit.hard_limit);
                
                if config.auto_terminate_on_violation {
                    stats.forced_terminations += 1;
                    // 在实际实现中，这里会终止相关任务
                    warn!("资源违规，触发自动终止");
                }
            }
        }
        
        Ok(!violation_detected)
    }
    
    /// 获取当前资源使用情况（模拟实现）
    fn get_current_resource_usage(&self, resource_type: &ResourceType) -> Result<u64> {
        // 这里是模拟实现，实际应该调用系统API获取真实数据
        match resource_type {
            ResourceType::Memory => {
                // 模拟内存使用
                Ok(512 * 1024 * 1024) // 512MB
            }
            ResourceType::CPU => {
                // 模拟CPU使用率
                Ok(45) // 45%
            }
            ResourceType::DiskRead => Ok(1024 * 1024), // 1MB
            ResourceType::DiskWrite => Ok(512 * 1024), // 512KB
            ResourceType::NetworkReceive => Ok(256 * 1024), // 256KB
            ResourceType::NetworkSend => Ok(128 * 1024), // 128KB
            ResourceType::GPU => Ok(0), // 未使用GPU
            ResourceType::GPUMemory => Ok(0),
            ResourceType::FileDescriptors => Ok(50),
            ResourceType::ThreadCount => Ok(8),
        }
    }
    
    /// 更新使用情况记录
    fn update_usage_record(&self, resource_type: ResourceType, current_usage: u64) -> Result<()> {
        let mut usage_map = self.current_usage.write().unwrap();
        let now = Utc::now();
        
        match usage_map.get_mut(&resource_type) {
            Some(usage) => {
                usage.current_usage = current_usage;
                if current_usage > usage.peak_usage {
                    usage.peak_usage = current_usage;
                }
                usage.sample_count += 1;
                usage.average_usage = (usage.average_usage * (usage.sample_count - 1) as f64 + current_usage as f64) / usage.sample_count as f64;
                usage.last_updated = now;
            }
            None => {
                let usage = ResourceUsage {
                    resource_type: resource_type.clone(),
                    current_usage,
                    peak_usage: current_usage,
                    average_usage: current_usage as f64,
                    last_updated: now,
                    sample_count: 1,
                };
                usage_map.insert(resource_type, usage);
            }
        }
        
        Ok(())
    }
    
    /// 添加事件到历史记录
    fn add_event(&self, event: ResourceMonitoringEvent) -> Result<()> {
        let mut history = self.event_history.write().unwrap();
        
        // 限制事件数量
        let config = self.config.read().unwrap();
        if history.len() >= config.max_events {
            history.remove(0); // 移除最旧的事件
        }
        
        history.push(event);
        Ok(())
    }
    
    /// 获取资源类型名称
    fn resource_type_name(&self, resource_type: &ResourceType) -> &'static str {
        match resource_type {
            ResourceType::CPU => "CPU",
            ResourceType::Memory => "内存",
            ResourceType::DiskRead => "磁盘读取",
            ResourceType::DiskWrite => "磁盘写入",
            ResourceType::NetworkReceive => "网络接收",
            ResourceType::NetworkSend => "网络发送",
            ResourceType::GPU => "GPU",
            ResourceType::GPUMemory => "GPU内存",
            ResourceType::FileDescriptors => "文件描述符",
            ResourceType::ThreadCount => "线程数",
        }
    }
    
    /// 获取统计信息
    pub fn get_statistics(&self) -> EnforcementStatistics {
        self.statistics.read().unwrap().clone()
    }
    
    /// 获取当前资源使用情况
    pub fn get_current_usage(&self) -> HashMap<ResourceType, ResourceUsage> {
        self.current_usage.read().unwrap().clone()
    }
    
    /// 获取事件历史
    pub fn get_event_history(&self) -> Vec<ResourceMonitoringEvent> {
        self.event_history.read().unwrap().clone()
    }
    
    /// 获取最近的事件
    pub fn get_recent_events(&self, limit: usize) -> Vec<ResourceMonitoringEvent> {
        let history = self.event_history.read().unwrap();
        let start_index = if history.len() > limit {
            history.len() - limit
        } else {
            0
        };
        history[start_index..].to_vec()
    }
    
    /// 清理旧事件
    pub fn cleanup_old_events(&self) -> Result<usize> {
        let config = self.config.read().unwrap();
        let retention_time = config.history_retention;
        let cutoff_time = Utc::now() - chrono::Duration::from_std(retention_time).unwrap();
        
        let mut history = self.event_history.write().unwrap();
        let initial_count = history.len();
        
        history.retain(|event| event.timestamp > cutoff_time);
        
        let removed_count = initial_count - history.len();
        Ok(removed_count)
    }
    
    /// 更新配置
    pub fn update_config(&self, new_config: ResourceEnforcerConfig) -> Result<()> {
        let mut config = self.config.write().unwrap();
        *config = new_config;
        debug!("更新资源强制执行器配置");
        Ok(())
    }
    
    /// 获取配置
    pub fn get_config(&self) -> ResourceEnforcerConfig {
        self.config.read().unwrap().clone()
    }

    /// 强制终止所有进程（用于安全清理）
    pub fn force_terminate_all_processes(&self) -> crate::Result<()> {
        info!("强制终止所有监控进程");
        
        // 停止所有监控活动
        if let Err(e) = self.stop_monitoring() {
            warn!("停止监控失败: {}", e);
        }
        
        // 清理所有注册的任务
        let mut tasks = self.registered_tasks.write().unwrap();
        let task_count = tasks.len();
        tasks.clear();
        
        // 更新统计信息
        let mut stats = self.statistics.write().unwrap();
        stats.forced_terminations += task_count as u64;
        
        // 生成终止事件
        let event = ResourceMonitoringEvent {
            event_id: Uuid::new_v4(),
            resource_type: ResourceType::ThreadCount,
            event_type: ResourceEventType::ResourceExhaustion,
            current_usage: 0,
            limit_value: 0,
            timestamp: Utc::now(),
            task_id: None,
            message: format!("强制终止了 {} 个监控进程", task_count),
            severity: EventSeverity::High,
        };
        
        if let Err(e) = self.add_event(event) {
            warn!("添加终止事件失败: {}", e);
        }
        
        info!("成功终止 {} 个监控进程", task_count);
        Ok(())
    }

    /// 获取活跃进程数量
    pub fn get_active_process_count(&self) -> usize {
        let tasks = self.registered_tasks.read().unwrap();
        let monitoring_active = *self.monitoring_active.lock().unwrap();
        
        if monitoring_active {
            tasks.len()
        } else {
            0
        }
    }
}

impl Default for ResourceEnforcer {
    fn default() -> Self {
        Self::new(ResourceEnforcerConfig::default())
    }
}

/// 创建严格的资源强制执行器
pub fn create_strict_resource_enforcer() -> ResourceEnforcer {
    let mut config = ResourceEnforcerConfig::default();
    
    // 设置更严格的限制
    config.auto_terminate_on_violation = true;
    config.warning_threshold = 0.7; // 70%时就警告
    config.monitoring_interval = Duration::from_millis(500); // 更频繁的监控
    
    // 更严格的内存限制
    config.resource_limits.insert(ResourceType::Memory, ResourceLimit {
        resource_type: ResourceType::Memory,
        soft_limit: 512 * 1024 * 1024, // 512MB
        hard_limit: 1024 * 1024 * 1024, // 1GB
        check_interval: Duration::from_secs(2),
        enabled: true,
    });
    
    // 更严格的CPU限制
    config.resource_limits.insert(ResourceType::CPU, ResourceLimit {
        resource_type: ResourceType::CPU,
        soft_limit: 60, // 60%
        hard_limit: 80, // 80%
        check_interval: Duration::from_secs(1),
        enabled: true,
    });
    
    // 添加线程数限制
    config.resource_limits.insert(ResourceType::ThreadCount, ResourceLimit {
        resource_type: ResourceType::ThreadCount,
        soft_limit: 10,
        hard_limit: 20,
        check_interval: Duration::from_secs(5),
        enabled: true,
    });
    
    // 添加文件描述符限制
    config.resource_limits.insert(ResourceType::FileDescriptors, ResourceLimit {
        resource_type: ResourceType::FileDescriptors,
        soft_limit: 100,
        hard_limit: 200,
        check_interval: Duration::from_secs(10),
        enabled: true,
    });
    
    ResourceEnforcer::new(config)
} 
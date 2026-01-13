use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use chrono::{DateTime, Utc};
use log::{info, error};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

use crate::error::{Error, Result};
use crate::types::{ResourceType, ResourceRequest, ResourceAllocation, TaskPriority};
use crate::resource::{ResourceCoordinator, ResourceMonitor, ResourceLimiter};
use crate::status::{StatusType, StatusTrackerTrait};

/// 资源分配策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// 优先级优先：基于任务优先级进行资源分配
    PriorityFirst,
    /// 公平分配：平均分配资源
    FairShare,
    /// 按需分配：基于需求量进行资源分配
    OnDemand,
    /// 预留分配：基于预留策略进行资源分配
    Reservation,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::PriorityFirst
    }
}

/// 缓存级别定义
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLevel {
    /// L1缓存：内存中的快速缓存
    L1,
    /// L2缓存：本地磁盘缓存
    L2,
    /// L3缓存：远程/分布式缓存
    L3,
}

/// 多级缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelCacheConfig {
    /// 各级缓存大小(字节)
    pub sizes: HashMap<CacheLevel, usize>,
    /// 各级缓存淘汰策略
    pub policies: HashMap<CacheLevel, crate::cache_common::EvictionPolicy>,
    /// 数据升降级阈值(访问次数)
    pub promotion_threshold: u64,
    /// 数据降级阈值(未访问时间，秒)
    pub demotion_threshold: u64,
    /// 预取策略是否启用
    pub prefetch_enabled: bool,
    /// 预取阈值(0.0-1.0)
    pub prefetch_threshold: f64,
}

impl Default for MultiLevelCacheConfig {
    fn default() -> Self {
        let mut sizes = HashMap::new();
        sizes.insert(CacheLevel::L1, 100 * 1024 * 1024); // 100MB
        sizes.insert(CacheLevel::L2, 1024 * 1024 * 1024); // 1GB
        sizes.insert(CacheLevel::L3, 10 * 1024 * 1024 * 1024); // 10GB
        
        let mut policies = HashMap::new();
        policies.insert(CacheLevel::L1, crate::cache_common::EvictionPolicy::LRU);
        policies.insert(CacheLevel::L2, crate::cache_common::EvictionPolicy::LFU);
        policies.insert(CacheLevel::L3, crate::cache_common::EvictionPolicy::FIFO);
        
        Self {
            sizes,
            policies,
            promotion_threshold: 5,
            demotion_threshold: 3600,
            prefetch_enabled: true,
            prefetch_threshold: 0.7,
        }
    }
}

/// 资源管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagerConfig {
    /// 资源分配策略
    pub allocation_strategy: AllocationStrategy,
    /// 最大并发任务数
    pub max_concurrent_tasks: usize,
    /// 资源预留比例(0.0-1.0)
    pub reservation_ratio: f64,
    /// 任务优先级权重(1-10)
    pub priority_weights: HashMap<TaskPriority, u32>,
    /// 监控间隔(毫秒)
    pub monitor_interval_ms: u64,
    /// 多级缓存配置
    pub cache_config: MultiLevelCacheConfig,
    /// 资源超额分配比例(0.0-1.0)
    pub overcommit_ratio: f64,
    /// 是否启用动态资源调整
    pub dynamic_adjustment: bool,
}

impl Default for ResourceManagerConfig {
    fn default() -> Self {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(TaskPriority::High, 10);
        priority_weights.insert(TaskPriority::Normal, 5);
        priority_weights.insert(TaskPriority::Low, 1);
        
        Self {
            allocation_strategy: AllocationStrategy::PriorityFirst,
            max_concurrent_tasks: 8,
            reservation_ratio: 0.2,
            priority_weights,
            monitor_interval_ms: 1000,
            cache_config: MultiLevelCacheConfig::default(),
            overcommit_ratio: 0.2,
            dynamic_adjustment: true,
        }
    }
}

/// 资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSummary {
    /// CPU使用率(百分比)
    pub cpu_usage_percent: f64,
    /// 内存使用量(字节)
    pub memory_usage_bytes: usize,
    /// 磁盘使用量(字节)
    pub disk_usage_bytes: usize,
    /// 网络使用量(字节/秒)
    pub network_usage_bytes_per_sec: usize,
    /// GPU使用率(如果有)
    pub gpu_usage_percent: Option<f64>,
    /// GPU内存使用量(如果有)
    pub gpu_memory_bytes: Option<usize>,
    /// 任务数量
    pub active_task_count: usize,
    /// 等待任务数量
    pub waiting_task_count: usize,
    /// 任务队列大小
    pub task_queue_size: usize,
    /// 总分配资源
    pub total_allocated: HashMap<ResourceType, usize>,
    /// 总可用资源
    pub total_available: HashMap<ResourceType, usize>,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
}

/// 资源使用趋势方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrendDirection {
    /// 增长趋势
    Increasing,
    /// 稳定趋势
    Stable,
    /// 下降趋势
    Decreasing,
}

/// 资源使用模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// 趋势方向
    pub trend: TrendDirection,
    /// 波动性
    pub volatility: f64,
    /// 高峰时间(小时)
    pub peak_times: Vec<u32>,
    /// 低谷时间(小时)
    pub idle_times: Vec<u32>,
}

/// 资源预测
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePredictions {
    /// CPU使用率预测
    pub cpu_usage_predicted: f64,
    /// 内存使用率预测
    pub memory_usage_predicted: f64,
    /// 推荐预留比例
    pub recommended_reservation: f64,
}

/// 资源使用模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsagePattern {
    /// CPU使用模式
    pub cpu: UsagePattern,
    /// 内存使用模式
    pub memory: UsagePattern,
    /// 资源预测
    pub predictions: ResourcePredictions,
}

/// 统一资源管理器
pub struct ResourceManager {
    /// 配置
    config: ResourceManagerConfig,
    /// 资源协调器
    coordinator: Arc<ResourceCoordinator>,
    /// 资源监控器
    monitor: Arc<ResourceMonitor>,
    /// 资源限制器
    limiter: Arc<ResourceLimiter>,
    /// 任务优先级队列
    priority_queue: Arc<Mutex<HashMap<TaskPriority, Vec<ResourceRequest>>>>,
    /// 资源使用摘要
    usage_summary: Arc<RwLock<ResourceUsageSummary>>,
    /// 任务资源使用映射
    task_resources: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    /// 状态追踪器
    status_tracker: Arc<dyn StatusTrackerTrait>,
    /// 命令发送器
    command_tx: Option<mpsc::Sender<ResourceCommand>>,
    /// 多级缓存命中率统计
    cache_stats: Arc<RwLock<HashMap<CacheLevel, (u64, u64)>>>,
    /// 是否已启动
    is_running: Arc<Mutex<bool>>,
}

/// 资源管理命令
enum ResourceCommand {
    /// 分配资源请求
    Allocate(ResourceRequest, mpsc::Sender<Result<ResourceAllocation>>),
    /// 检查资源分配
    CheckAllocation(ResourceRequest, mpsc::Sender<Result<bool>>),
    /// 释放资源
    Release(String, mpsc::Sender<Result<()>>),
    /// 更新优先级
    UpdatePriority(String, TaskPriority, mpsc::Sender<Result<()>>),
    /// 获取使用情况
    GetUsage(mpsc::Sender<Result<ResourceUsageSummary>>),
    /// 终止管理器
    Terminate,
}

impl ResourceManager {
    /// 获取资源管理器配置
    pub fn config(&self) -> &ResourceManagerConfig {
        &self.config
    }

    /// 创建新的资源管理器
    pub fn new(
        config: ResourceManagerConfig,
        coordinator: Arc<ResourceCoordinator>,
        status_tracker: Arc<dyn StatusTrackerTrait>,
    ) -> Self {
        let monitor = Arc::new(ResourceMonitor::new(Arc::clone(&coordinator), 100));
        let limiter = Arc::new(ResourceLimiter::new());
        
        // 初始化优先级队列
        let mut priority_queue = HashMap::new();
        priority_queue.insert(TaskPriority::High, Vec::new());
        priority_queue.insert(TaskPriority::Normal, Vec::new());
        priority_queue.insert(TaskPriority::Low, Vec::new());
        
        // 初始化使用摘要
        let usage_summary = ResourceUsageSummary {
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            disk_usage_bytes: 0,
            network_usage_bytes_per_sec: 0,
            gpu_usage_percent: None,
            gpu_memory_bytes: None,
            active_task_count: 0,
            waiting_task_count: 0,
            task_queue_size: 0,
            total_allocated: HashMap::new(),
            total_available: HashMap::new(),
            timestamp: Utc::now(),
        };
        
        // 初始化缓存统计
        let mut cache_stats = HashMap::new();
        cache_stats.insert(CacheLevel::L1, (0, 0));
        cache_stats.insert(CacheLevel::L2, (0, 0));
        cache_stats.insert(CacheLevel::L3, (0, 0));
        
        Self {
            config,
            coordinator,
            monitor,
            limiter,
            priority_queue: Arc::new(Mutex::new(priority_queue)),
            usage_summary: Arc::new(RwLock::new(usage_summary)),
            task_resources: Arc::new(RwLock::new(HashMap::new())),
            status_tracker,
            command_tx: None,
            cache_stats: Arc::new(RwLock::new(cache_stats)),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// 启动资源管理器
    pub async fn start(&mut self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| Error::Internal(format!("锁定失败: {}", e)))?;
        if *is_running {
            return Err(Error::InvalidState("资源管理器已经在运行".to_string()));
        }
        
        info!("启动资源管理器");
        
        // 创建命令通道
        let (tx, rx) = mpsc::channel::<ResourceCommand>(100);
        self.command_tx = Some(tx);
        
        // 设置运行状态
        *is_running = true;
        
        // 启动管理循环
        let config = self.config.clone();
        let coordinator = Arc::clone(&self.coordinator);
        let monitor = Arc::clone(&self.monitor);
        let limiter = Arc::clone(&self.limiter);
        let priority_queue = Arc::clone(&self.priority_queue);
        let usage_summary = Arc::clone(&self.usage_summary);
        let task_resources = Arc::clone(&self.task_resources);
        let status_tracker = Arc::clone(&self.status_tracker);
        let is_running_clone = Arc::clone(&self.is_running);
        
        tokio::spawn(async move {
            Self::run_manager_loop(
                rx,
                config,
                coordinator,
                monitor,
                limiter,
                priority_queue,
                usage_summary,
                task_resources,
                status_tracker,
                is_running_clone,
            ).await;
        });
        
        info!("资源管理器启动完成");
        Ok(())
    }

    /// 停止资源管理器
    pub async fn stop(&self) -> Result<()> {
        if let Some(tx) = &self.command_tx {
            tx.send(ResourceCommand::Terminate).await
                .map_err(|e| Error::Internal(format!("发送终止命令失败: {}", e)))?;
        }
        
        let mut is_running = self.is_running.lock().map_err(|e| Error::Internal(format!("锁定失败: {}", e)))?;
        *is_running = false;
        
        info!("资源管理器已停止");
        Ok(())
    }
    
    /// 请求资源分配
    pub async fn request_resource(&self, request: ResourceRequest) -> Result<ResourceAllocation> {
        if let Some(tx) = &self.command_tx {
            let (resp_tx, mut resp_rx) = mpsc::channel(1);
            tx.send(ResourceCommand::Allocate(request.clone(), resp_tx)).await
                .map_err(|_| Error::Internal("资源管理通道已关闭".into()))?;
            
            resp_rx.recv().await.ok_or_else(|| Error::Internal("资源管理响应通道已关闭".into()))?
        } else {
            Err(Error::Internal("资源管理器未启动".into()))
        }
    }
    
    /// 检查是否可以分配资源
    // 训练任务相关功能已移除 - 向量数据库系统不需要训练功能
    pub async fn can_allocate_resources(&self, _task_id: &crate::compat::task_scheduler::core::TaskId) -> Result<bool> {
        // 训练任务资源分配功能已移除
        // 向量数据库系统不需要训练任务资源管理
        Ok(false)
    }
    
    /// 释放资源
    pub async fn release_resource(&self, allocation_id: &str) -> Result<()> {
        if let Some(tx) = &self.command_tx {
            let (resp_tx, mut resp_rx) = mpsc::channel(1);
            tx.send(ResourceCommand::Release(allocation_id.to_string(), resp_tx)).await
                .map_err(|_| Error::Internal("资源管理通道已关闭".into()))?;
            
            resp_rx.recv().await.ok_or_else(|| Error::Internal("资源管理响应通道已关闭".into()))?
        } else {
            Err(Error::Internal("资源管理器未启动".into()))
        }
    }
    
    /// 更新任务优先级
    pub async fn update_task_priority(&self, task_id: &str, priority: TaskPriority) -> Result<()> {
        if let Some(tx) = &self.command_tx {
            let (resp_tx, mut resp_rx) = mpsc::channel(1);
            tx.send(ResourceCommand::UpdatePriority(task_id.to_string(), priority, resp_tx)).await
                .map_err(|_| Error::Internal("资源管理通道已关闭".into()))?;
            
            resp_rx.recv().await.ok_or_else(|| Error::Internal("资源管理响应通道已关闭".into()))?
        } else {
            Err(Error::Internal("资源管理器未启动".into()))
        }
    }
    
    /// 获取资源使用情况
    pub async fn get_resource_usage(&self) -> Result<ResourceUsageSummary> {
        if let Some(tx) = &self.command_tx {
            let (resp_tx, mut resp_rx) = mpsc::channel(1);
            tx.send(ResourceCommand::GetUsage(resp_tx)).await
                .map_err(|_| Error::Internal("资源管理通道已关闭".into()))?;
            
            resp_rx.recv().await.ok_or_else(|| Error::Internal("资源管理响应通道已关闭".into()))?
        } else {
            // 直接从摘要中读取
            let summary = self.usage_summary.read().unwrap().clone();
            Ok(summary)
        }
    }
    
    /// 获取缓存命中率
    pub fn get_cache_hit_ratio(&self, level: CacheLevel) -> f64 {
        let stats = self.cache_stats.read().unwrap();
        if let Some((hits, total)) = stats.get(&level) {
            if *total == 0 {
                0.0
            } else {
                *hits as f64 / *total as f64
            }
        } else {
            0.0
        }
    }
    
    /// 记录缓存访问
    pub fn record_cache_access(&self, level: CacheLevel, hit: bool) {
        let mut stats = self.cache_stats.write().unwrap();
        if let Some((hits, total)) = stats.get_mut(&level) {
            *total += 1;
            if hit {
                *hits += 1;
            }
        }
    }

    /// 分配资源
    pub async fn allocate_resources(&self, requests: &[ResourceRequest]) -> Result<ResourceAllocation> {
        let allocation = ResourceAllocation {
            allocation_id: uuid::Uuid::new_v4().to_string(),
            request_id: requests.first().map(|r| r.requester_id.clone()).unwrap_or_default(),
            resource_type: requests.first().map(|r| r.resource_type.clone()).unwrap_or_default(),
            amount: requests.iter().map(|r| r.amount).sum(),
            allocated_at: Utc::now(),
            expires_at: None,
            resource_location: None,
            metadata: HashMap::new(),
        };

        // 简化的资源分配逻辑 - 只处理第一个请求的资源类型
        // 在实际实现中，这里应该处理多个不同类型的资源请求

        // 记录分配
        let mut task_resources = self.task_resources.write().unwrap();
        task_resources.insert(allocation.request_id.clone(), allocation.clone());

        Ok(allocation)
    }

    /// 释放资源
    pub async fn release_resources(&self, allocation: &ResourceAllocation) -> Result<()> {
        let mut task_resources = self.task_resources.write().unwrap();
        task_resources.remove(&allocation.request_id);
        Ok(())
    }
    
    /// 运行资源管理循环
    async fn run_manager_loop(
        mut rx: mpsc::Receiver<ResourceCommand>,
        config: ResourceManagerConfig,
        coordinator: Arc<ResourceCoordinator>,
        _monitor: Arc<ResourceMonitor>,
        limiter: Arc<ResourceLimiter>,
        priority_queue: Arc<Mutex<HashMap<TaskPriority, Vec<ResourceRequest>>>>,
        usage_summary: Arc<RwLock<ResourceUsageSummary>>,
        task_resources: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
        status_tracker: Arc<dyn StatusTrackerTrait>,
        is_running: Arc<Mutex<bool>>,
    ) {
        while let Some(cmd) = rx.recv().await {
            match cmd {
                ResourceCommand::Allocate(request, resp_tx) => {
                    let result = Self::handle_allocation_request(
                        &config,
                        &coordinator,
                        &limiter,
                        &priority_queue,
                        &task_resources,
                        status_tracker.as_ref(),
                        request,
                    ).await;
                    
                    let _ = resp_tx.send(result).await;
                },
                ResourceCommand::CheckAllocation(request, resp_tx) => {
                    let result = Self::handle_check_allocation_request(
                        &config,
                        &coordinator,
                        &limiter,
                        &priority_queue,
                        &task_resources,
                        status_tracker.as_ref(),
                        request,
                    ).await;
                    
                    let _ = resp_tx.send(result).await;
                },
                ResourceCommand::Release(allocation_id, resp_tx) => {
                    let result = Self::handle_release_request(
                        &coordinator,
                        &task_resources,
                        &allocation_id,
                    ).await;
                    
                    let _ = resp_tx.send(result).await;
                },
                ResourceCommand::UpdatePriority(task_id, priority, resp_tx) => {
                    let result = Self::handle_priority_update(
                        &priority_queue,
                        &task_id,
                        priority,
                    ).await;
                    
                    let _ = resp_tx.send(result).await;
                },
                ResourceCommand::GetUsage(resp_tx) => {
                    let summary = usage_summary.read().unwrap().clone();
                    let _ = resp_tx.send(Ok(summary)).await;
                },
                ResourceCommand::Terminate => {
                    // 设置运行标志
                    {
                        let mut running = is_running.lock().unwrap();
                        *running = false;
                    }
                    break;
                },
            }
        }
    }
    
    /// 处理资源分配请求
    async fn handle_allocation_request(
        config: &ResourceManagerConfig,
        coordinator: &ResourceCoordinator,
        _limiter: &ResourceLimiter,
        priority_queue: &Arc<Mutex<HashMap<TaskPriority, Vec<ResourceRequest>>>>,
        task_resources: &Arc<RwLock<HashMap<String, ResourceAllocation>>>,
        status_tracker: &dyn StatusTrackerTrait,
        request: ResourceRequest,
    ) -> Result<ResourceAllocation> {
        // 根据分配策略处理请求
        match config.allocation_strategy {
            AllocationStrategy::PriorityFirst => {
                // 检查资源是否可用
                let available = coordinator.get_resource_usage()?;
                let resource_type = &request.resource_type;
                // 统一用ResourceUsageSummary的total_allocated/total_available
                let used = available.get(resource_type).map(|(used, _)| *used).unwrap_or(0);
                let total = available.get(resource_type).map(|(_, total)| *total).unwrap_or(0);
                let remaining = total.saturating_sub(used);
                if remaining >= request.amount {
                    // 资源足够，直接分配
                    let allocation = coordinator.request_resource(request.clone()).await?;
                    if let Some(_task_id) = &request.task_id {
                        let mut resources = task_resources.write().unwrap();
                        resources.insert(allocation.allocation_id.clone(), allocation.clone());
                    }
                    return Ok(allocation);
                } else {
                    // 资源不足，根据优先级决定是否等待
                    if request.can_wait {
                        {
                            let mut queue = priority_queue.lock().unwrap();
                            let priority_queue = queue.entry(request.priority).or_insert_with(Vec::new);
                            priority_queue.push(request.clone());
                        }
                        if let Some(task_id) = &request.task_id {
                            if let Ok(uuid) = uuid::Uuid::parse_str(task_id) {
                                let event = crate::status::StatusEvent::new(
                                    uuid,
                                    StatusType::Waiting,
                                    format!("等待资源: {:?}", resource_type),
                                );
                                status_tracker.update_status(event).await?;
                            }
                        }
                        return Err(Error::ResourceUnavailable(format!(
                            "资源不足，请求已加入等待队列: {:?}", resource_type
                        )));
                    } else {
                        return Err(Error::ResourceUnavailable(format!(
                            "资源不足: {:?}", resource_type
                        )));
                    }
                }
            },
            AllocationStrategy::FairShare => {
                // 公平分配实现
                // 根据活动任务数量平均分配资源
                let allocation = coordinator.request_resource(request.clone()).await?;
                
                // 记录任务资源分配
                if let Some(_task_id) = &request.task_id {
                    let mut resources = task_resources.write().unwrap();
                    resources.insert(allocation.allocation_id.clone(), allocation.clone());
                }
                
                Ok(allocation)
            },
            AllocationStrategy::OnDemand => {
                // 按需分配实现
                let allocation = coordinator.request_resource(request.clone()).await?;
                
                // 记录任务资源分配
                if let Some(_task_id) = &request.task_id {
                    let mut resources = task_resources.write().unwrap();
                    resources.insert(allocation.allocation_id.clone(), allocation.clone());
                }
                
                Ok(allocation)
            },
            AllocationStrategy::Reservation => {
                // 预留分配实现
                // 确保高优先级任务始终有预留资源
                let allocation = coordinator.request_resource(request.clone()).await?;
                
                // 记录任务资源分配
                if let Some(_task_id) = &request.task_id {
                    let mut resources = task_resources.write().unwrap();
                    resources.insert(allocation.allocation_id.clone(), allocation.clone());
                }
                
                Ok(allocation)
            },
        }
    }
    
    /// 处理资源释放请求
    async fn handle_release_request(
        coordinator: &ResourceCoordinator,
        task_resources: &Arc<RwLock<HashMap<String, ResourceAllocation>>>,
        allocation_id: &str,
    ) -> Result<()> {
        // 释放资源
        coordinator.release_resource(allocation_id).await?;
        
        // 从记录中移除
        {
            let mut resources = task_resources.write().unwrap();
            resources.remove(allocation_id);
        }
        
        Ok(())
    }
    
    /// 处理优先级更新
    async fn handle_priority_update(
        priority_queue: &Arc<Mutex<HashMap<TaskPriority, Vec<ResourceRequest>>>>,
        task_id: &str,
        new_priority: TaskPriority,
    ) -> Result<()> {
        let mut queue = priority_queue.lock().unwrap();
        
        // 找到任务并更新优先级
        let mut found = false;
        let mut request_to_move = None;
        
        // 遍历所有优先级队列
        for (priority, requests) in queue.iter_mut() {
            if *priority == new_priority {
                continue;
            }
            
            // 在当前优先级队列中查找任务
            let mut i = 0;
            while i < requests.len() {
                if let Some(req_task_id) = &requests[i].task_id {
                    if req_task_id == task_id {
                        // 找到任务，移除并准备移动
                        let mut request = requests.remove(i);
                        request.priority = new_priority;
                        request_to_move = Some(request);
                        found = true;
                        break;
                    }
                }
                i += 1;
            }
            
            if found {
                break;
            }
        }
        
        // 如果找到了请求，将其移动到新的优先级队列
        if let Some(request) = request_to_move {
            let new_queue = queue.entry(new_priority).or_insert_with(Vec::new);
            new_queue.push(request);
            Ok(())
        } else {
            Err(Error::NotFound(format!("未找到任务 {} 的资源请求", task_id)))
        }
    }
    
    /// 更新资源使用情况摘要
    async fn update_resource_usage(
        usage_summary: &Arc<RwLock<ResourceUsageSummary>>,
        monitor: &ResourceMonitor,
        coordinator: &ResourceCoordinator,
    ) {
        if let Ok(resource_usage) = coordinator.get_resource_usage() {
            let mut summary = usage_summary.write().unwrap();
            // 直接赋值，不再解构HashMap
            summary.cpu_usage_percent = if let Ok(usage) = monitor.get_current_usage() {
                usage.get(&ResourceType::CPU).cloned().unwrap_or(0.0) * 100.0
            } else { 0.0 };
            summary.memory_usage_bytes = resource_usage.get(&ResourceType::Memory).map(|(used, _)| *used).unwrap_or(0);
            summary.disk_usage_bytes = resource_usage.get(&ResourceType::Storage).map(|(used, _)| *used).unwrap_or(0);
            // 直接使用resource_usage HashMap创建总分配和总可用映射
            summary.total_allocated = resource_usage.iter().map(|(k, (used, _))| (k.clone(), *used)).collect();
            summary.total_available = resource_usage.iter().map(|(k, (_, total))| (k.clone(), *total)).collect();
            summary.timestamp = Utc::now();
        }
    }

    /// 执行智能资源分配
    pub async fn optimize_resource_allocation(&self) -> Result<()> {
        // 获取当前资源使用情况
        let usage = self.get_resource_usage().await?;
        
        // 检查是否有等待中的任务
        let has_waiting_tasks = {
            let priority_queues = self.priority_queue.lock().unwrap();
            priority_queues.iter().any(|(_, queue)| !queue.is_empty())
        };
        
        // 如果没有等待中的任务，不需要优化
        if !has_waiting_tasks {
            return Ok(());
        }
        
        // 创建队列副本并释放锁
        let mut priority_queues_copy = {
            let priority_queues = self.priority_queue.lock().unwrap();
            priority_queues.clone()
        };
        
        // 根据优先级和历史使用情况进行资源优化
        self.execute_intelligent_allocation(&mut priority_queues_copy, &usage).await?;
        
        // 更新原队列
        {
            let mut priority_queues = self.priority_queue.lock().unwrap();
            *priority_queues = priority_queues_copy;
        }
        
        Ok(())
    }
    
    /// 执行智能资源分配算法
    async fn execute_intelligent_allocation(
        &self,
        priority_queues: &mut HashMap<TaskPriority, Vec<ResourceRequest>>,
        current_usage: &ResourceUsageSummary,
    ) -> Result<()> {
        let reservation_ratio = self.config.reservation_ratio;
        let mut allocatable_resources = HashMap::new();
        for (resource_type, allocated) in &current_usage.total_allocated {
            let used = *allocated;
            let capacity = current_usage.total_available.get(resource_type).cloned().unwrap_or(0);
            let remaining = capacity.saturating_sub(used);
            let high_priority_reserved = ((capacity as f64) * reservation_ratio) as usize;
            allocatable_resources.insert(resource_type.clone(), (remaining, high_priority_reserved));
        }
        
        // 3. 分类待处理的任务请求
        let mut high_priority_requests = Vec::new();
        let mut medium_priority_requests = Vec::new();
        let mut low_priority_requests = Vec::new();
        
        // 3.1 收集高优先级请求
        if let Some(high_queue) = priority_queues.get(&TaskPriority::High) {
            high_priority_requests.extend(high_queue.clone());
        }
        
        // 3.2 收集中优先级请求
        if let Some(medium_queue) = priority_queues.get(&TaskPriority::Normal) {
            medium_priority_requests.extend(medium_queue.clone());
        }
        
        // 3.3 收集低优先级请求
        if let Some(low_queue) = priority_queues.get(&TaskPriority::Low) {
            low_priority_requests.extend(low_queue.clone());
        }
        
        // 4. 按优先级处理请求
        
        // 4.1 处理高优先级请求
        let high_results = self.process_priority_batch(
            &high_priority_requests, 
            &mut allocatable_resources, 
            true
        ).await?;
        
        // 4.2 处理中优先级请求
        let medium_results = self.process_priority_batch(
            &medium_priority_requests, 
            &mut allocatable_resources, 
            false
        ).await?;
        
        // 4.3 处理低优先级请求
        let low_results = self.process_priority_batch(
            &low_priority_requests, 
            &mut allocatable_resources, 
            false
        ).await?;
        
        // 5. 更新队列
        
        // 5.1 更新高优先级队列
        if let Some(high_queue) = priority_queues.get_mut(&TaskPriority::High) {
            high_queue.clear();
            for (request, processed) in high_priority_requests.iter().zip(high_results.iter()) {
                if !processed {
                    high_queue.push(request.clone());
                }
            }
        }
        
        // 5.2 更新中优先级队列
        if let Some(medium_queue) = priority_queues.get_mut(&TaskPriority::Normal) {
            medium_queue.clear();
            for (request, processed) in medium_priority_requests.iter().zip(medium_results.iter()) {
                if !processed {
                    medium_queue.push(request.clone());
                }
            }
        }
        
        // 5.3 更新低优先级队列
        if let Some(low_queue) = priority_queues.get_mut(&TaskPriority::Low) {
            low_queue.clear();
            for (request, processed) in low_priority_requests.iter().zip(low_results.iter()) {
                if !processed {
                    low_queue.push(request.clone());
                }
            }
        }
        
        Ok(())
    }
    
    /// 处理一批相同优先级的请求
    async fn process_priority_batch(
        &self,
        requests: &[ResourceRequest],
        allocatable_resources: &mut HashMap<ResourceType, (usize, usize)>,
        can_use_reserved: bool,
    ) -> Result<Vec<bool>> {
        let mut results = vec![false; requests.len()];
        
        for (i, request) in requests.iter().enumerate() {
            let resource_type = &request.resource_type;
            
            // 跳过已处理的请求
            if results[i] {
                continue;
            }
            
            // 检查是否有足够资源
            if let Some((remaining, reserved)) = allocatable_resources.get_mut(resource_type) {
                let required = request.amount;
                
                // 检查资源是否足够
                let can_allocate = if can_use_reserved {
                    // 高优先级可以使用预留资源
                    *remaining >= required
                } else {
                    // 中低优先级不能使用预留资源
                    *remaining >= required && (*remaining - required) >= *reserved
                };
                
                if can_allocate {
                    // 尝试分配资源
                    match self.coordinator.request_resource(request.clone()).await {
                        Ok(allocation) => {
                            // 分配成功，更新可用资源
                            *remaining = remaining.saturating_sub(required);
                            
                            // 记录任务资源分配
                            if let Some(_task_id) = &request.task_id {
                                let mut resources = self.task_resources.write().unwrap();
                                resources.insert(allocation.allocation_id.clone(), allocation);
                            }
                            
                            // 标记为已处理
                            results[i] = true;
                        },
                        Err(_) => {
                            // 分配失败，跳过
                            continue;
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// 启动资源优化循环
    pub async fn start_optimization(self: Arc<Self>) -> Result<()> {
        // 获取当前实例的引用
        let self_arc = Arc::clone(&self);
        
        // 启动优化循环
        let manager = Arc::clone(&self.coordinator);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // 检查是否仍在运行
                {
                    let running = self_arc.is_running.lock().unwrap();
                    if !*running {
                        break;
                    }
                }
                
                // 执行资源优化
                if let Err(e) = manager.optimize_resources().await {
                    error!("资源优化失败: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// 分析资源使用历史
    pub async fn analyze_resource_usage_history(&self) -> Result<ResourceUsagePattern> {
        // 从监控器获取历史资源使用数据
        let cpu_trends = self.monitor.get_resource_trends(&ResourceType::CPU, 24)?;
        let memory_trends = self.monitor.get_resource_trends(&ResourceType::Memory, 24)?;
        
        // 分析CPU使用模式
        let cpu_pattern = self.analyze_resource_pattern(&cpu_trends);
        
        // 分析内存使用模式
        let memory_pattern = self.analyze_resource_pattern(&memory_trends);
        
        // 返回资源使用模式
        Ok(ResourceUsagePattern {
            cpu: cpu_pattern,
            memory: memory_pattern,
            predictions: self.predict_future_needs().await?,
        })
    }
    
    /// 分析资源使用模式
    fn analyze_resource_pattern(&self, trends: &[(DateTime<Utc>, f64)]) -> UsagePattern {
        // 至少需要6个数据点
        if trends.len() < 6 {
            return UsagePattern {
                trend: TrendDirection::Stable,
                volatility: 0.0,
                peak_times: Vec::new(),
                idle_times: Vec::new(),
            };
        }
        
        // 计算平均值和标准差
        let values: Vec<f64> = trends.iter().map(|(_, v)| *v).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // 计算趋势方向
        let first_half_avg = values.iter().take(values.len() / 2).sum::<f64>() / (values.len() / 2) as f64;
        let second_half_avg = values.iter().skip(values.len() / 2).sum::<f64>() / (values.len() - values.len() / 2) as f64;
        
        let trend = if (second_half_avg - first_half_avg).abs() < 0.1 {
            TrendDirection::Stable
        } else if second_half_avg > first_half_avg {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };
        
        // 计算波动性
        let volatility = std_dev / mean;
        
        // 识别峰值和低谷时间
        let threshold_high = mean + std_dev * 0.5;
        let threshold_low = mean - std_dev * 0.5;
        
        let mut peak_times = Vec::new();
        let mut idle_times = Vec::new();
        
        for (time, value) in trends {
            let hour = time.time().hour() as u32;
            
            if *value > threshold_high {
                if !peak_times.contains(&hour) {
                    peak_times.push(hour);
                }
            } else if *value < threshold_low {
                if !idle_times.contains(&hour) {
                    idle_times.push(hour);
                }
            }
        }
        
        UsagePattern {
            trend,
            volatility,
            peak_times,
            idle_times,
        }
    }
    
    /// 预测未来资源需求
    async fn predict_future_needs(&self) -> Result<ResourcePredictions> {
        // 使用监控器的预测功能
        let cpu_prediction = self.monitor.predict_future_usage(&ResourceType::CPU, 2)?;
        let memory_prediction = self.monitor.predict_future_usage(&ResourceType::Memory, 2)?;
        
        Ok(ResourcePredictions {
            cpu_usage_predicted: cpu_prediction,
            memory_usage_predicted: memory_prediction,
            recommended_reservation: self.calculate_recommended_reservation(cpu_prediction, memory_prediction),
        })
    }
    
    /// 计算推荐预留量
    fn calculate_recommended_reservation(&self, cpu_prediction: f64, memory_prediction: f64) -> f64 {
        // 预留比例基于预测使用率
        // 使用率越高，预留比例越大
        let base_reservation = 0.1; // 基础预留10%
        let dynamic_factor = (cpu_prediction + memory_prediction) / 2.0; // 动态因子
        
        // 最终预留比例: 10% - 30%
        base_reservation + dynamic_factor * 0.2
    }

    /// 兼容外部调用的资源优化方法
    pub async fn optimize_resources(&self) -> Result<()> {
        self.optimize_resource_allocation().await
    }
    
    /// 处理资源分配请求（内部实现，避免与上方同名冲突）
    async fn handle_allocation_request_internal(
        _config: &ResourceManagerConfig,
        coordinator: &ResourceCoordinator,
        _limiter: &ResourceLimiter,
        _priority_queue: &Arc<Mutex<HashMap<TaskPriority, Vec<ResourceRequest>>>>,
        task_resources: &Arc<RwLock<HashMap<String, ResourceAllocation>>>,
        _status_tracker: &dyn crate::status::StatusTrackerTrait,
        request: ResourceRequest,
    ) -> Result<ResourceAllocation> {
        // 尝试分配资源
        let allocation = coordinator.request_resource(request.clone()).await?;
        
        // 记录任务资源分配
        if let Some(_task_id) = &request.task_id {
            let mut resources = task_resources.write().unwrap();
            resources.insert(allocation.allocation_id.clone(), allocation.clone());
        }
        
        Ok(allocation)
    }
    
    /// 处理资源分配检查请求
    async fn handle_check_allocation_request(
        _config: &ResourceManagerConfig,
        coordinator: &ResourceCoordinator,
        _limiter: &ResourceLimiter,
        _priority_queue: &Arc<Mutex<HashMap<TaskPriority, Vec<ResourceRequest>>>>,
        _task_resources: &Arc<RwLock<HashMap<String, ResourceAllocation>>>,
        _status_tracker: &dyn crate::status::StatusTrackerTrait,
        request: ResourceRequest,
    ) -> Result<bool> {
        // 检查是否可以分配资源
        // 兼容：ResourceCoordinator 未提供显式可用性检查接口，改为尝试零影响分配判断
        // 方案：读取可用资源映射，比较请求量；若不足则返回 false
        let available = coordinator.get_resource_usage()?; // (allocated, total)
        let maybe = available.get(&request.resource_type).cloned();
        if let Some((allocated, total)) = maybe {
            Ok(total >= allocated + request.amount)
        } else {
            Ok(false)
        }
    }
} 
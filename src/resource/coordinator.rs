//! 资源协调器模块
//! 
//! 提供高级的资源管理功能，包括协调器、限制器和监控器

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use tokio::sync::Mutex as AsyncMutex;

use crate::error::{Error, Result};
use crate::status::{StatusTracker, StatusTrackerTrait};
use crate::event::enhanced::{EnhancedEventSystem, DomainEvent};
use super::{ResourceType, ResourceRequest, ResourceAllocation, ResourcePool, InternalResourceType};
use log;

/// 资源协调器 - 管理系统资源分配与回收
pub struct ResourceCoordinator {
    /// 可用资源池
    available_resources: RwLock<HashMap<ResourceType, usize>>,
    /// 已分配的资源
    allocated_resources: RwLock<HashMap<String, ResourceAllocation>>,
    /// 等待资源的请求队列
    waiting_requests: AsyncMutex<VecDeque<(ResourceRequest, tokio::sync::oneshot::Sender<Result<ResourceAllocation>>)>>,
    /// 资源分配通知
    notification: tokio::sync::Notify,
    /// 状态追踪器
    status_tracker: Arc<dyn StatusTrackerTrait>,
    /// 事件系统
    event_system: Option<Arc<EnhancedEventSystem>>,
    /// 资源池集合
    pools: Arc<AsyncMutex<HashMap<ResourceType, ResourcePool>>>,
}

impl ResourceCoordinator {
    /// 创建新的资源协调器
    pub fn new(status_tracker: Arc<StatusTracker>) -> Self {
        let mut available_resources = HashMap::new();
        
        // 初始化默认资源
        available_resources.insert(ResourceType::CPU, num_cpus::get());
        available_resources.insert(ResourceType::Memory, 1024 * 1024 * 1024); // 1GB，单位：KB
        available_resources.insert(ResourceType::Storage, 10 * 1024 * 1024 * 1024); // 10GB，单位：KB
        
        Self {
            available_resources: RwLock::new(available_resources),
            allocated_resources: RwLock::new(HashMap::new()),
            waiting_requests: AsyncMutex::new(VecDeque::new()),
            notification: tokio::sync::Notify::new(),
            status_tracker,
            event_system: None,
            pools: Arc::new(AsyncMutex::new(HashMap::new())),
        }
    }
    
    /// 设置事件系统
    pub fn with_event_system(mut self, event_system: Arc<EnhancedEventSystem>) -> Self {
        self.event_system = Some(event_system);
        self
    }
    
    /// 启动资源协调器
    pub async fn start(&self) -> Result<()> {
        // 注册事件处理器
        if let Some(event_system) = &self.event_system {
            // 示例：注册一个资源申请事件处理器
            event_system.register_handler("resource_request", Box::new(|event| {
                println!("收到资源请求事件: {:?}", event);
                Ok(())
            }))?;
        }
        
        // 发布协调器启动事件
        self.publish_event("resource_coordinator_started", "ResourceCoordinator启动")?;
        
        // 启动任务处理循环
        let self_arc = Arc::new(self.clone());
        tokio::spawn(async move {
            self_arc.process_waiting_requests().await;
        });
        
        Ok(())
    }

    /// 注册资源池
    pub async fn register_pool(&self, resource_type: ResourceType, capacity: u64) -> Result<()> {
        let mut pools = self.pools.lock().await;
        
        if pools.contains_key(&resource_type) {
            return Err(Error::AlreadyExists(format!("资源池 {:?} 已存在", resource_type)));
        }

        // 将ResourceType转换为InternalResourceType
        let internal_resource_type = match &resource_type {
            ResourceType::CPU => InternalResourceType::CPU,
            ResourceType::Memory => InternalResourceType::Memory,
            ResourceType::Storage => InternalResourceType::Storage,
            ResourceType::Network => InternalResourceType::NetworkBandwidth,
            ResourceType::GPU => InternalResourceType::GPU,
            ResourceType::AlgorithmExecution => InternalResourceType::Custom(0),
            ResourceType::Custom(name) => {
                // 使用名称的哈希值作为自定义资源ID
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                name.hash(&mut hasher);
                InternalResourceType::Custom(hasher.finish() as u32)
            }
        };
        
        let pool = ResourcePool::new(internal_resource_type, capacity);
        pools.insert(resource_type, pool);

        Ok(())
    }
    
    /// 请求资源分配
    pub async fn request_resource(&self, request: ResourceRequest) -> Result<ResourceAllocation> {
        let resource_type = request.resource_type.clone();
        let amount = request.amount;
        
        // 发布事件
        let mut event_data = HashMap::new();
        event_data.insert("resource_type".to_string(), format!("{:?}", resource_type));
        event_data.insert("amount".to_string(), amount.to_string());
        event_data.insert("requester_id".to_string(), request.requester_id.clone());
        
        self.publish_event_with_data("resource_request", "收到资源请求", event_data)?;
        
        // 检查资源是否足够
        let can_allocate = {
            let available = self.available_resources.read().unwrap();
            match available.get(&resource_type) {
                Some(available_amount) => *available_amount >= amount,
                None => false,
            }
        };
        
        if can_allocate {
            let allocation = self.allocate_resource(request).await?;
            
            // 发布分配成功事件
            self.publish_event("resource_allocated", &format!("资源 {:?} 分配成功: {}", resource_type, amount))?;
            
            Ok(allocation)
        } else {
            // 资源不足，加入等待队列
            let (sender, receiver) = tokio::sync::oneshot::channel();
            {
                let mut waiting = self.waiting_requests.lock().await;
                waiting.push_back((request, sender));
            }
            
            // 发布等待事件
            self.publish_event("resource_waiting", &format!("资源请求等待中: {:?} 数量 {}", resource_type, amount))?;
            
            // 等待分配结果
            receiver.await.map_err(|_| Error::resource("等待资源分配超时".into()))?
        }
    }
    
    /// 释放已分配的资源
    pub async fn release_resource(&self, allocation_id: &str) -> Result<()> {
        let allocation = {
            let mut allocated = self.allocated_resources.write().unwrap();
            match allocated.remove(allocation_id) {
                Some(a) => a,
                None => return Err(Error::NotFound(format!("未找到资源分配: {}", allocation_id))),
            }
        };
        
        // 归还资源到可用池
        {
            let mut available = self.available_resources.write().unwrap();
            let current = available.entry(allocation.resource_type.clone()).or_insert(0);
            *current += allocation.amount;
        }
        
        // 发布资源释放事件
        let mut event_data = HashMap::new();
        event_data.insert("resource_type".to_string(), format!("{:?}", allocation.resource_type));
        event_data.insert("amount".to_string(), allocation.amount.to_string());
        event_data.insert("allocation_id".to_string(), allocation_id.to_string());
        
        self.publish_event_with_data("resource_released", "资源已释放", event_data)?;
        
        // 通知等待的请求
        self.notification.notify_one();
        
        Ok(())
    }
    
    /// 获取当前资源使用情况
    pub fn get_resource_usage(&self) -> Result<HashMap<ResourceType, (usize, usize)>> {
        let available = self.available_resources.read().unwrap();
        let allocated = self.allocated_resources.read().unwrap();
        
        let mut result = HashMap::new();
        let mut total_allocated: HashMap<ResourceType, usize> = HashMap::new();
        
        // 计算每种资源类型的已分配总量
        for allocation in allocated.values() {
            let entry = total_allocated.entry(allocation.resource_type.clone()).or_insert(0);
            *entry += allocation.amount;
        }
        
        // 计算每种资源的总量和已用量
        for (res_type, available_amount) in available.iter() {
            let allocated_amount = total_allocated.get(res_type).cloned().unwrap_or(0);
            let total_amount = allocated_amount + available_amount;
            result.insert(res_type.clone(), (allocated_amount, total_amount));
        }
        
        Ok(result)
    }
    
    /// 内部方法：分配资源
    async fn allocate_resource(&self, request: ResourceRequest) -> Result<ResourceAllocation> {
        let resource_type = request.resource_type.clone();
        let amount = request.amount;
        
        // 检查资源是否足够
        let can_allocate = {
            let available = self.available_resources.read().unwrap();
            match available.get(&resource_type) {
                Some(available_amount) => *available_amount >= amount,
                None => false,
            }
        };
        
        if can_allocate {
            self.do_allocate_resource(request).await
        } else {
            Err(Error::ResourceUnavailable(format!(
                "资源 {:?} 不足",
                resource_type
            )))
        }
    }
    
    /// 实际执行资源分配的内部方法
    async fn do_allocate_resource(&self, request: ResourceRequest) -> Result<ResourceAllocation> {
        let allocation_id = uuid::Uuid::new_v4().to_string();
        let resource_type = request.resource_type.clone();
        let amount = request.amount;
        
        // 扣减可用资源
        {
            let mut available = self.available_resources.write().unwrap();
            if let Some(available_amount) = available.get_mut(&resource_type) {
                *available_amount -= amount;
            }
        }
        
        // 记录分配
                        let allocation = ResourceAllocation {
                    allocation_id: allocation_id.clone(),
                    request_id: request.id.clone(),
                    resource_type: resource_type.clone(),
                    amount,
                    allocated_at: chrono::Utc::now(),
                    expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(24)),
                    resource_location: Some(format!("pool_{}", resource_type.to_string().to_lowercase())),
                    metadata: HashMap::new(),
                };
        
        {
            let mut allocated = self.allocated_resources.write().unwrap();
            allocated.insert(allocation_id.clone(), allocation.clone());
        }
        
        // 发布分配事件
        self.publish_event("resource_allocated", &format!("已分配资源 {:?}: {}", resource_type, amount))?;
        
        Ok(allocation)
    }
    
    /// 内部方法：处理等待的资源请求
    async fn process_waiting_requests(&self) {
        loop {
            self.notification.notified().await;
            
            let mut waiting = self.waiting_requests.lock().await;
            
            let mut i = 0;
            while i < waiting.len() {
                let (request, _) = &waiting[i];
                
                // 检查资源是否足够
                let can_allocate = {
                    let available = self.available_resources.read().unwrap();
                    match available.get(&request.resource_type) {
                        Some(available_amount) => *available_amount >= request.amount,
                        None => false,
                    }
                };
                
                if can_allocate {
                    if let Some((request, sender)) = waiting.remove(i) {
                        match self.try_allocate_now(&request).await {
                            Ok(allocation) => {
                                if sender.send(Ok(allocation)).is_err() {
                                    // 接收方已关闭，可能已超时
                                }
                            },
                            Err(e) => {
                                if sender.send(Err(e)).is_err() {
                                    // 接收方已关闭
                                }
                            }
                        }
                    }
                } else {
                    i += 1;
                }
            }
        }
    }
    
    /// 内部方法：尝试立即分配资源
    async fn try_allocate_now(&self, request: &ResourceRequest) -> Result<ResourceAllocation> {
        let resource_type = &request.resource_type;
        let amount = request.amount;
        
        // 再次检查并分配
        let mut available = self.available_resources.write().unwrap();
        if let Some(available_amount) = available.get_mut(resource_type) {
            if *available_amount >= amount {
                *available_amount -= amount;
                
                // 释放写锁后继续
                drop(available);
                
                let allocation_id = uuid::Uuid::new_v4().to_string();
                let allocation = ResourceAllocation {
                    allocation_id: allocation_id.clone(),
                    request_id: request.id.clone(),
                    resource_type: resource_type.clone(),
                    amount,
                    allocated_at: chrono::Utc::now(),
                    expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(24)),
                    resource_location: Some(format!("pool_{}", resource_type.to_string().to_lowercase())),
                    metadata: HashMap::new(),
                };
                
                self.allocated_resources
                    .write()
                    .unwrap()
                    .insert(allocation_id.clone(), allocation.clone());
                
                return Ok(allocation);
            }
        }
        
        Err(Error::ResourceUnavailable(format!(
            "资源 {:?} 仍然不足",
            resource_type
        )))
    }
    
    /// 设置可用资源
    pub fn set_available_resource(&self, resource_type: ResourceType, amount: usize) -> Result<()> {
        let mut available = self.available_resources.write().unwrap();
        available.insert(resource_type, amount);
        Ok(())
    }
    
    /// 发布标准事件
    fn publish_event(&self, event_name: &str, message: &str) -> Result<()> {
        if let Some(event_system) = &self.event_system {
            let domain_event = DomainEvent::new(event_name.to_string(), "ResourceCoordinator".to_string())
                .with_data("message", message);
            
            event_system.publish_domain_event(&domain_event)?;
        }
        Ok(())
    }
    
    /// 发布带数据的事件
    fn publish_event_with_data(&self, event_name: &str, message: &str, data: HashMap<String, String>) -> Result<()> {
        if let Some(event_system) = &self.event_system {
            let mut domain_event = DomainEvent::new(event_name.to_string(), "ResourceCoordinator".to_string())
                .with_data("message", message);
            
            for (key, value) in data {
                domain_event = domain_event.add_data(key, value);
            }
            
            event_system.publish_domain_event(&domain_event)?;
        }
        Ok(())
    }

    /// 优化资源分配
    pub async fn optimize_resources(&self) -> Result<()> {
        // 清理过期的分配
        self.cleanup_expired_allocations().await?;
        
        // 重新平衡资源池
        self.rebalance_resource_pools().await?;
        
        // 优化等待队列
        self.optimize_waiting_queue().await?;
        
        // 发布优化完成事件
        self.publish_event("resource_optimization_completed", "资源优化已完成")?;
        
        Ok(())
    }

    /// 清理过期的分配
    async fn cleanup_expired_allocations(&self) -> Result<()> {
        let mut expired_allocations = Vec::new();
        
        // 收集过期的分配
        {
            let allocated = self.allocated_resources.read().unwrap();
            for (allocation_id, allocation) in allocated.iter() {
                if allocation.is_expired() {
                    expired_allocations.push(allocation_id.clone());
                }
            }
        }
        
        // 释放过期的分配
        for allocation_id in expired_allocations {
            if let Err(e) = self.release_resource(&allocation_id).await {
                log::warn!("清理过期分配 {} 失败: {}", allocation_id, e);
            }
        }
        
        Ok(())
    }

    /// 重新平衡资源池
    async fn rebalance_resource_pools(&self) -> Result<()> {
        let mut pools = self.pools.lock().await;
        
        for (resource_type, pool) in pools.iter_mut() {
            // 清理过期的分配
            pool.cleanup_expired();
            
            // 检查池的利用率
            let utilization = pool.utilization();
            
            // 如果利用率过低，可以考虑缩减容量
            if utilization < 0.3 && pool.total_capacity > 100 {
                // 在实际实现中，这里可以动态调整容量
                log::info!("资源池 {:?} 利用率较低: {:.2}%", resource_type, utilization * 100.0);
            }
            
            // 如果利用率过高，可能需要扩容
            else if utilization > 0.9 {
                log::warn!("资源池 {:?} 利用率过高: {:.2}%", resource_type, utilization * 100.0);
            }
        }
        
        Ok(())
    }

    /// 优化等待队列
    async fn optimize_waiting_queue(&self) -> Result<()> {
        let mut waiting = self.waiting_requests.lock().await;
        
        if waiting.is_empty() {
            return Ok(());
        }
        
        // 按优先级排序等待队列
        waiting.make_contiguous().sort_by(|a, b| {
            let priority_a = &a.0.priority;
            let priority_b = &b.0.priority;
            
            // 高优先级在前
            priority_b.cmp(priority_a)
        });
        
        // 尝试满足高优先级请求
        let mut processed_indices = Vec::new();
        
        for (index, (request, _)) in waiting.iter().enumerate() {
            // 检查资源是否足够
            let can_allocate = {
                let available = self.available_resources.read().unwrap();
                match available.get(&request.resource_type) {
                    Some(available_amount) => *available_amount >= request.amount,
                    None => false,
                }
            };
            
            if can_allocate {
                processed_indices.push(index);
                // 限制每次处理的数量，避免阻塞
                if processed_indices.len() >= 5 {
                    break;
                }
            }
        }
        
        // 从后往前删除已处理的请求，避免索引偏移
        for &index in processed_indices.iter().rev() {
            if let Some((request, sender)) = waiting.remove(index) {
                match self.try_allocate_now(&request).await {
                    Ok(allocation) => {
                        if sender.send(Ok(allocation)).is_err() {
                            log::warn!("发送分配结果失败，接收方可能已关闭");
                        }
                    },
                    Err(e) => {
                        if sender.send(Err(e)).is_err() {
                            log::warn!("发送分配错误失败，接收方可能已关闭");
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// 获取资源统计信息
    pub fn get_resource_statistics(&self) -> Result<ResourceStatistics> {
        let available = self.available_resources.read().unwrap();
        let allocated = self.allocated_resources.read().unwrap();
        
        let mut total_allocated = HashMap::new();
        let mut total_available = HashMap::new();
        
        // 统计已分配资源
        for allocation in allocated.values() {
            let entry = total_allocated.entry(allocation.resource_type.clone()).or_insert(0);
            *entry += allocation.amount;
        }
        
        // 统计可用资源
        for (resource_type, amount) in available.iter() {
            total_available.insert(resource_type.clone(), *amount);
        }
        
        Ok(ResourceStatistics {
            total_allocated,
            total_available,
            active_allocations: allocated.len(),
            waiting_requests: self.waiting_requests.try_lock().map(|w| w.len()).unwrap_or(0),
            last_updated: Utc::now(),
        })
    }
}

impl Clone for ResourceCoordinator {
    fn clone(&self) -> Self {
        Self {
            available_resources: RwLock::new(self.available_resources.read().unwrap().clone()),
            allocated_resources: RwLock::new(self.allocated_resources.read().unwrap().clone()),
            waiting_requests: AsyncMutex::new(VecDeque::new()),
            notification: tokio::sync::Notify::new(),
            status_tracker: self.status_tracker.clone(),
            event_system: self.event_system.clone(),
            pools: self.pools.clone(),
        }
    }
}

/// 资源限制器
pub struct ResourceLimiter {
    limits: RwLock<HashMap<ResourceType, usize>>,
    usage: RwLock<HashMap<String, HashMap<ResourceType, usize>>>,
}

impl ResourceLimiter {
    /// 创建新的资源限制器
    pub fn new() -> Self {
        let mut limits = HashMap::new();
        
        // 设置默认限制
        limits.insert(ResourceType::CPU, num_cpus::get());
        limits.insert(ResourceType::Memory, 1024 * 1024 * 1024); // 1GB
        limits.insert(ResourceType::Storage, 10 * 1024 * 1024 * 1024); // 10GB
        
        Self {
            limits: RwLock::new(limits),
            usage: RwLock::new(HashMap::new()),
        }
    }
    
    /// 设置资源限制
    pub fn set_limit(&self, resource_type: ResourceType, limit: usize) {
        let mut limits = self.limits.write().unwrap();
        limits.insert(resource_type, limit);
    }
    
    /// 检查资源限制
    pub fn check_limit(&self, operation_id: &str, resource_type: &ResourceType, amount: usize) -> Result<()> {
        let limits = self.limits.read().unwrap();
        let usage = self.usage.read().unwrap();
        
        let limit = limits.get(resource_type).copied().unwrap_or(usize::MAX);
        let current_usage = usage.get(operation_id)
            .and_then(|u| u.get(resource_type))
            .copied()
            .unwrap_or(0);
        
        if current_usage + amount > limit {
            return Err(Error::ResourceUnavailable(format!(
                "操作 {} 的资源 {:?} 使用量 {} + {} 超过限制 {}",
                operation_id, resource_type, current_usage, amount, limit
            )));
        }
        
        Ok(())
    }
    
    /// 记录资源使用
    pub fn record_usage(&self, operation_id: &str, resource_type: ResourceType, amount: usize) {
        let mut usage = self.usage.write().unwrap();
        let op_usage = usage.entry(operation_id.to_string()).or_insert_with(HashMap::new);
        let current = op_usage.entry(resource_type).or_insert(0);
        *current += amount;
    }
    
    /// 释放资源使用记录
    pub fn release_usage(&self, operation_id: &str) {
        let mut usage = self.usage.write().unwrap();
        usage.remove(operation_id);
    }
}

/// 资源监控器
pub struct ResourceMonitor {
    metrics: RwLock<HashMap<ResourceType, Vec<(DateTime<Utc>, f64)>>>,
    coordinator: Arc<ResourceCoordinator>,
    samples_to_keep: usize,
}

impl ResourceMonitor {
    /// 创建新的资源监控器
    pub fn new(coordinator: Arc<ResourceCoordinator>, samples_to_keep: usize) -> Self {
        Self {
            metrics: RwLock::new(HashMap::new()),
            coordinator,
            samples_to_keep: samples_to_keep.max(10), // 至少保留10个样本
        }
    }
    
    /// 启动监控
    pub async fn start(&self) -> Result<()> {
        let self_arc = Arc::new(self.clone());
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                if let Err(e) = self_arc.collect_metrics().await {
                    eprintln!("收集指标失败: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// 收集指标
    async fn collect_metrics(&self) -> Result<()> {
        let usage = self.coordinator.get_resource_usage()?;
        let now = Utc::now();
        
        let mut metrics = self.metrics.write().unwrap();
        
        for (resource_type, (used, total)) in usage {
            let utilization = if total > 0 {
                used as f64 / total as f64
            } else {
                0.0
            };
            
            let type_metrics = metrics.entry(resource_type).or_insert_with(Vec::new);
            type_metrics.push((now, utilization));
            
            // 保持指定数量的样本
            if type_metrics.len() > self.samples_to_keep {
                type_metrics.remove(0);
            }
        }
        
        Ok(())
    }
    
    /// 获取资源趋势
    pub fn get_resource_trends(&self, resource_type: &ResourceType, points: usize) -> Result<Vec<(DateTime<Utc>, f64)>> {
        let metrics = self.metrics.read().unwrap();
        
        if let Some(type_metrics) = metrics.get(resource_type) {
            let start_index = if type_metrics.len() > points {
                type_metrics.len() - points
            } else {
                0
            };
            
            Ok(type_metrics[start_index..].to_vec())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// 获取当前使用率
    pub fn get_current_usage(&self) -> Result<HashMap<ResourceType, f64>> {
        let metrics = self.metrics.read().unwrap();
        let mut result = HashMap::new();
        
        for (resource_type, type_metrics) in metrics.iter() {
            if let Some((_, utilization)) = type_metrics.last() {
                result.insert(resource_type.clone(), *utilization);
            }
        }
        
        Ok(result)
    }
    
    /// 预测未来使用量
    pub fn predict_future_usage(&self, resource_type: &ResourceType, hours_ahead: usize) -> Result<f64> {
        let metrics = self.metrics.read().unwrap();
        
        if let Some(type_metrics) = metrics.get(resource_type) {
            if type_metrics.len() < 2 {
                return Ok(0.0);
            }
            
            // 简单的线性预测
            let recent_count = (type_metrics.len() / 2).max(2);
            let recent_metrics = &type_metrics[type_metrics.len() - recent_count..];
            
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;
            
            for (i, (_, utilization)) in recent_metrics.iter().enumerate() {
                let x = i as f64;
                let y = *utilization;
                
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }
            
            let n = recent_metrics.len() as f64;
            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;
            
            let future_x = n + hours_ahead as f64;
            let predicted = slope * future_x + intercept;
            
            Ok(predicted.max(0.0).min(1.0)) // 限制在 0-1 之间
        } else {
            Ok(0.0)
        }
    }
}

impl Clone for ResourceMonitor {
    fn clone(&self) -> Self {
        Self {
            metrics: RwLock::new(self.metrics.read().unwrap().clone()),
            coordinator: self.coordinator.clone(),
            samples_to_keep: self.samples_to_keep,
        }
    }
}

/// 资源统计信息
#[derive(Debug, Clone)]
pub struct ResourceStatistics {
    /// 已分配资源总量
    pub total_allocated: HashMap<ResourceType, usize>,
    /// 可用资源总量
    pub total_available: HashMap<ResourceType, usize>,
    /// 活跃分配数量
    pub active_allocations: usize,
    /// 等待请求数量
    pub waiting_requests: usize,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
}
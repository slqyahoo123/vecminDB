/// 状态同步模块
/// 
/// 提供跨模块的状态一致性管理，确保各模块状态同步

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::Duration;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, error, debug, warn};
use crate::Result;

use crate::core::{CoreEvent, EventBusInterface};

/// 同步策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncStrategy {
    Immediate,    // 立即同步
    Batched,      // 批量同步
    Scheduled,    // 定时同步
    EventDriven,  // 事件驱动同步
}

/// 状态同步状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncStatus {
    Idle,
    Synchronizing,
    Completed,
    Failed,
    Conflict,
}

/// 状态快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub id: String,
    pub module_id: String,
    pub state_type: String,
    pub state_data: HashMap<String, String>,
    pub version: u64,
    pub checksum: String,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl StateSnapshot {
    pub fn new(module_id: String, state_type: String, state_data: HashMap<String, String>) -> Self {
        let checksum = Self::calculate_checksum(&state_data);
        
        Self {
            id: Uuid::new_v4().to_string(),
            module_id,
            state_type,
            state_data,
            version: 1,
            checksum,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_version(mut self, version: u64) -> Self {
        self.version = version;
        self.checksum = Self::calculate_checksum(&self.state_data);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// 计算状态数据的校验和
    fn calculate_checksum(state_data: &HashMap<String, String>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // 对键进行排序，确保校验和的一致性
        let mut sorted_items: Vec<_> = state_data.iter().collect();
        sorted_items.sort_by_key(|(k, _)| *k);
        
        for (key, value) in sorted_items {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        
        format!("{:x}", hasher.finish())
    }
    
    /// 验证校验和
    pub fn verify_checksum(&self) -> bool {
        let calculated_checksum = Self::calculate_checksum(&self.state_data);
        calculated_checksum == self.checksum
    }
}

/// 同步任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncTask {
    pub id: String,
    pub source_module: String,
    pub target_modules: Vec<String>,
    pub state_type: String,
    pub strategy: SyncStrategy,
    pub status: SyncStatus,
    pub source_snapshot: Option<StateSnapshot>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub retry_count: usize,
    pub max_retries: usize,
    pub error_message: Option<String>,
}

impl SyncTask {
    pub fn new(
        source_module: String,
        target_modules: Vec<String>,
        state_type: String,
        strategy: SyncStrategy,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_module,
            target_modules,
            state_type,
            strategy,
            status: SyncStatus::Idle,
            source_snapshot: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
        }
    }
    
    pub fn with_source_snapshot(mut self, snapshot: StateSnapshot) -> Self {
        self.source_snapshot = Some(snapshot);
        self
    }
    
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}

/// 状态提供者接口
#[async_trait]
pub trait StateProvider: Send + Sync {
    /// 模块ID
    fn module_id(&self) -> &str;
    
    /// 获取状态快照
    async fn get_state_snapshot(&self, state_type: &str) -> Result<StateSnapshot>;
    
    /// 获取所有状态类型
    async fn get_state_types(&self) -> Result<Vec<String>>;
    
    /// 检查状态是否发生变化
    async fn has_state_changed(&self, state_type: &str, last_version: u64) -> Result<bool>;
}

/// 状态应用者接口
#[async_trait]
pub trait StateApplier: Send + Sync {
    /// 模块ID
    fn module_id(&self) -> &str;
    
    /// 应用状态快照
    async fn apply_state_snapshot(&self, snapshot: StateSnapshot) -> Result<()>;
    
    /// 验证状态快照是否可应用
    async fn validate_state_snapshot(&self, snapshot: &StateSnapshot) -> Result<bool>;
    
    /// 解决状态冲突
    async fn resolve_state_conflict(
        &self,
        local_snapshot: StateSnapshot,
        remote_snapshot: StateSnapshot,
    ) -> Result<StateSnapshot>;
}

/// 状态同步管理器
pub struct StateSyncManager {
    sync_interval: Duration,
    event_bus: Arc<dyn EventBusInterface>,
    
    // 状态提供者和应用者
    state_providers: Arc<RwLock<HashMap<String, Arc<dyn StateProvider>>>>,
    state_appliers: Arc<RwLock<HashMap<String, Arc<dyn StateApplier>>>>,
    
    // 同步任务管理
    active_tasks: Arc<RwLock<HashMap<String, SyncTask>>>,
    state_cache: Arc<RwLock<HashMap<String, StateSnapshot>>>,
    
    // 配置
    batch_size: usize,
    conflict_resolution_strategy: ConflictResolutionStrategy,
}

/// 冲突解决策略
#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    SourceWins,      // 源优先
    TargetWins,      // 目标优先
    LatestWins,      // 最新优先
    ManualResolve,   // 手动解决
}

impl StateSyncManager {
    pub fn new(
        sync_interval_seconds: u64,
        event_bus: Arc<dyn EventBusInterface>,
    ) -> Result<Self> {
        Ok(Self {
            sync_interval: Duration::from_secs(sync_interval_seconds),
            event_bus,
            state_providers: Arc::new(RwLock::new(HashMap::new())),
            state_appliers: Arc::new(RwLock::new(HashMap::new())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            state_cache: Arc::new(RwLock::new(HashMap::new())),
            batch_size: 10,
            conflict_resolution_strategy: ConflictResolutionStrategy::LatestWins,
        })
    }
    
    /// 注册状态提供者
    pub fn register_state_provider(&self, provider: Arc<dyn StateProvider>) {
        let provider_id = provider.module_id().to_string();
        let mut providers = self.state_providers.write().unwrap();
        providers.insert(provider_id.clone(), provider);
        info!("注册状态提供者: {}", provider_id);
    }
    
    /// 注册状态应用者
    pub fn register_state_applier(&self, applier: Arc<dyn StateApplier>) {
        let applier_id = applier.module_id().to_string();
        let mut appliers = self.state_appliers.write().unwrap();
        appliers.insert(applier_id.clone(), applier);
        info!("注册状态应用者: {}", applier_id);
    }
    
    /// 启动状态同步管理器
    pub async fn start(&self) -> Result<()> {
        info!("启动状态同步管理器...");
        
        // 启动定时同步任务
        self.start_periodic_sync().await?;
        
        // 注册事件监听器
        self.register_event_listeners().await?;
        
        info!("状态同步管理器启动完成");
        Ok(())
    }
    
    /// 启动定时同步任务
    async fn start_periodic_sync(&self) -> Result<()> {
        let manager = Arc::new(self.clone_ref());
        let sync_interval = self.sync_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(sync_interval);
            loop {
                interval.tick().await;
                if let Err(e) = manager.perform_scheduled_sync().await {
                    error!("定时同步失败: {}", e);
                }
            }
        });
        
        info!("定时同步任务已启动，间隔: {:?}", sync_interval);
        Ok(())
    }
    
    /// 注册事件监听器
    async fn register_event_listeners(&self) -> Result<()> {
        // 监听状态变化事件
        let state_change_handler = StateChangeEventHandler {
            sync_manager: self.create_weak_ref(),
        };
        self.event_bus.subscribe("state.changed", Box::new(state_change_handler)).await?;
        
        Ok(())
    }
    
    /// 执行定时同步
    async fn perform_scheduled_sync(&self) -> Result<()> {
        debug!("执行定时状态同步");
        
        let providers = self.state_providers.read().unwrap().clone();
        
        // 为每个状态提供者创建同步任务
        for (module_id, provider) in &providers {
            match provider.get_state_types().await {
                Ok(state_types) => {
                    for state_type in state_types {
                        // 检查状态是否发生变化
                        let last_version = self.get_cached_version(module_id, &state_type);
                        
                        match provider.has_state_changed(&state_type, last_version).await {
                            Ok(true) => {
                                // 创建同步任务
                                let target_modules = self.get_target_modules_for_state(
                                    module_id, 
                                    &state_type
                                );
                                
                                if !target_modules.is_empty() {
                                    let task = SyncTask::new(
                                        module_id.clone(),
                                        target_modules,
                                        state_type,
                                        SyncStrategy::Scheduled,
                                    );
                                    
                                    self.execute_sync_task(task).await?;
                                }
                            },
                            Ok(false) => {
                                // 状态未变化，跳过
                            },
                            Err(e) => {
                                warn!("检查状态变化失败: {} -> {}: {}", module_id, state_type, e);
                            }
                        }
                    }
                },
                Err(e) => {
                    warn!("获取状态类型失败: {}: {}", module_id, e);
                }
            }
        }
        
        Ok(())
    }
    
    /// 执行同步任务
    pub async fn execute_sync_task(&self, mut task: SyncTask) -> Result<()> {
        let task_id = task.id.clone();
        info!("执行同步任务: {} ({} -> {:?})", 
              task_id, task.source_module, task.target_modules);
        
        // 更新任务状态
        task.status = SyncStatus::Synchronizing;
        task.started_at = Some(Utc::now());
        
        // 保存任务
        {
            let mut tasks = self.active_tasks.write().unwrap();
            tasks.insert(task_id.clone(), task.clone());
        }
        
        // 发送同步开始事件
        self.send_sync_event(&task, "sync.started").await?;
        
        // 执行同步
        match self.perform_sync(&mut task).await {
            Ok(_) => {
                // 同步成功
                task.status = SyncStatus::Completed;
                task.completed_at = Some(Utc::now());
                
                // 更新任务
                {
                    let mut tasks = self.active_tasks.write().unwrap();
                    tasks.insert(task_id.clone(), task.clone());
                }
                
                // 发送同步完成事件
                self.send_sync_event(&task, "sync.completed").await?;
                
                info!("同步任务完成: {}", task_id);
                Ok(())
            },
            Err(e) => {
                // 同步失败
                task.status = SyncStatus::Failed;
                task.error_message = Some(e.to_string());
                task.completed_at = Some(Utc::now());
                
                // 更新任务
                {
                    let mut tasks = self.active_tasks.write().unwrap();
                    tasks.insert(task_id.clone(), task.clone());
                }
                
                // 发送同步失败事件
                self.send_sync_event(&task, "sync.failed").await?;
                
                error!("同步任务失败: {}: {}", task_id, e);
                
                // 如果还有重试次数，则重试
                if task.retry_count < task.max_retries {
                    task.retry_count += 1;
                    task.status = SyncStatus::Idle;
                    tokio::time::sleep(Duration::from_secs(5)).await; // 等待5秒后重试
                    return Box::pin(self.execute_sync_task(task)).await;
                }
                
                Err(e)
            }
        }
    }
    
    /// 执行实际的同步操作
    async fn perform_sync(&self, task: &mut SyncTask) -> Result<()> {
        // 获取源状态快照，避免跨await边界持有锁
        let source_snapshot = {
            let provider = {
                let providers = self.state_providers.read().unwrap();
                providers.get(&task.source_module).map(|p| p.clone())
            };
            
            if let Some(provider) = provider {
                provider.get_state_snapshot(&task.state_type).await?
            } else {
                return Err(crate::Error::Internal(
                    format!("状态提供者不存在: {}", task.source_module)
                ));
            }
        };
        
        task.source_snapshot = Some(source_snapshot.clone());
        
        // 缓存状态快照
        self.cache_state_snapshot(source_snapshot.clone());
        
        // 向所有目标模块应用状态
        let appliers = self.state_appliers.read().unwrap().clone();
        
        for target_module in &task.target_modules {
            if let Some(applier) = appliers.get(target_module) {
                // 验证状态快照是否可应用
                match applier.validate_state_snapshot(&source_snapshot).await {
                    Ok(true) => {
                        // 应用状态快照
                        match applier.apply_state_snapshot(source_snapshot.clone()).await {
                            Ok(_) => {
                                debug!("状态同步成功: {} -> {}", task.source_module, target_module);
                            },
                            Err(e) => {
                                warn!("状态应用失败: {} -> {}: {}", 
                                      task.source_module, target_module, e);
                            }
                        }
                    },
                    Ok(false) => {
                        warn!("状态快照验证失败: {} -> {}", task.source_module, target_module);
                    },
                    Err(e) => {
                        warn!("状态快照验证错误: {} -> {}: {}", 
                              task.source_module, target_module, e);
                    }
                }
            } else {
                warn!("状态应用者不存在: {}", target_module);
            }
        }
        
        Ok(())
    }
    
    /// 缓存状态快照
    fn cache_state_snapshot(&self, snapshot: StateSnapshot) {
        let cache_key = format!("{}:{}", snapshot.module_id, snapshot.state_type);
        let mut cache = self.state_cache.write().unwrap();
        cache.insert(cache_key, snapshot);
    }
    
    /// 获取缓存的版本号
    fn get_cached_version(&self, module_id: &str, state_type: &str) -> u64 {
        let cache_key = format!("{}:{}", module_id, state_type);
        let cache = self.state_cache.read().unwrap();
        cache.get(&cache_key).map(|snapshot| snapshot.version).unwrap_or(0)
    }
    
    /// 获取状态的目标模块列表
    fn get_target_modules_for_state(&self, source_module: &str, state_type: &str) -> Vec<String> {
        // 简化实现：返回所有其他模块作为目标
        let appliers = self.state_appliers.read().unwrap();
        appliers.keys()
            .filter(|&module_id| module_id != source_module)
            .cloned()
            .collect()
    }
    
    /// 发送同步事件
    async fn send_sync_event(&self, task: &SyncTask, event_type: &str) -> Result<()> {
        let event = CoreEvent::new(event_type, "state_sync_manager")
            .with_data("task_id", &task.id)
            .with_data("source_module", &task.source_module)
            .with_data("state_type", &task.state_type)
            .with_data("status", &format!("{:?}", task.status));
            
        self.event_bus.publish(&event).await?;
        Ok(())
    }
    
    /// 获取同步任务状态
    pub fn get_sync_task_status(&self, task_id: &str) -> Option<SyncStatus> {
        let tasks = self.active_tasks.read().unwrap();
        tasks.get(task_id).map(|task| task.status.clone())
    }
    
    /// 列出活跃的同步任务
    pub fn list_active_sync_tasks(&self) -> Vec<String> {
        let tasks = self.active_tasks.read().unwrap();
        tasks.keys().cloned().collect()
    }
    
    /// 创建引用副本（简化实现）
    fn clone_ref(&self) -> Self {
        Self {
            sync_interval: self.sync_interval,
            event_bus: self.event_bus.clone(),
            state_providers: self.state_providers.clone(),
            state_appliers: self.state_appliers.clone(),
            active_tasks: self.active_tasks.clone(),
            state_cache: self.state_cache.clone(),
            batch_size: self.batch_size,
            conflict_resolution_strategy: self.conflict_resolution_strategy.clone(),
        }
    }
    
    /// 创建弱引用（简化实现）
    fn create_weak_ref(&self) -> StateSyncManagerRef {
        StateSyncManagerRef {
            manager_id: "state_sync_manager".to_string(),
        }
    }
}

/// 状态同步管理器引用（简化版）
#[derive(Clone)]
pub struct StateSyncManagerRef {
    pub manager_id: String,
}

/// 状态变化事件处理器
pub struct StateChangeEventHandler {
    pub sync_manager: StateSyncManagerRef,
}

#[async_trait]
impl crate::core::EventHandler for StateChangeEventHandler {
    async fn handle(&self, event: &CoreEvent) -> Result<()> {
        debug!("处理状态变化事件: {}", event.event_type);
        
        // 从事件中提取模块ID和状态类型
        if let (Some(module_id), Some(state_type)) = (
            event.data.get("module_id"),
            event.data.get("state_type"),
        ) {
            info!("检测到状态变化: {} -> {}", module_id, state_type);
            
            // 实际实现中会触发立即同步
            // 这里简化为日志记录
        }
        
        Ok(())
    }
}

/// 简单状态提供者示例
pub struct SimpleStateProvider {
    module_id: String,
    state_data: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
    version_counter: Arc<RwLock<u64>>,
}

impl SimpleStateProvider {
    pub fn new(module_id: String) -> Self {
        Self {
            module_id,
            state_data: Arc::new(RwLock::new(HashMap::new())),
            version_counter: Arc::new(RwLock::new(1)),
        }
    }
    
    /// 更新状态数据
    pub fn update_state(&self, state_type: String, data: HashMap<String, String>) {
        {
            let mut state_data = self.state_data.write().unwrap();
            state_data.insert(state_type, data);
        }
        
        // 增加版本号
        {
            let mut version = self.version_counter.write().unwrap();
            *version += 1;
        }
    }
}

#[async_trait]
impl StateProvider for SimpleStateProvider {
    fn module_id(&self) -> &str {
        &self.module_id
    }
    
    async fn get_state_snapshot(&self, state_type: &str) -> Result<StateSnapshot> {
        let state_data = self.state_data.read().unwrap();
        let version = *self.version_counter.read().unwrap();
        
        if let Some(data) = state_data.get(state_type) {
            Ok(StateSnapshot::new(
                self.module_id.clone(),
                state_type.to_string(),
                data.clone(),
            ).with_version(version))
        } else {
            Err(crate::Error::NotFound(format!("状态类型不存在: {}", state_type)))
        }
    }
    
    async fn get_state_types(&self) -> Result<Vec<String>> {
        let state_data = self.state_data.read().unwrap();
        Ok(state_data.keys().cloned().collect())
    }
    
    async fn has_state_changed(&self, _state_type: &str, last_version: u64) -> Result<bool> {
        let current_version = *self.version_counter.read().unwrap();
        Ok(current_version > last_version)
    }
}

/// 简单状态应用者示例
pub struct SimpleStateApplier {
    module_id: String,
    applied_states: Arc<RwLock<HashMap<String, StateSnapshot>>>,
}

impl SimpleStateApplier {
    pub fn new(module_id: String) -> Self {
        Self {
            module_id,
            applied_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl StateApplier for SimpleStateApplier {
    fn module_id(&self) -> &str {
        &self.module_id
    }
    
    async fn apply_state_snapshot(&self, snapshot: StateSnapshot) -> Result<()> {
        // 验证校验和
        if !snapshot.verify_checksum() {
            return Err(crate::Error::Validation("状态快照校验和验证失败".to_string()));
        }
        
        // 应用状态
        let state_type = snapshot.state_type.clone();
        let mut applied_states = self.applied_states.write().unwrap();
        applied_states.insert(state_type.clone(), snapshot);
        
        debug!("状态快照已应用: {} -> {}", self.module_id, state_type);
        Ok(())
    }
    
    async fn validate_state_snapshot(&self, snapshot: &StateSnapshot) -> Result<bool> {
        // 简单验证：检查校验和
        Ok(snapshot.verify_checksum())
    }
    
    async fn resolve_state_conflict(
        &self,
        local_snapshot: StateSnapshot,
        remote_snapshot: StateSnapshot,
    ) -> Result<StateSnapshot> {
        // 简单策略：选择版本号更高的
        if remote_snapshot.version > local_snapshot.version {
            Ok(remote_snapshot)
        } else {
            Ok(local_snapshot)
        }
    }
} 
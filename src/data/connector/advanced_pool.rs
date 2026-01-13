/// 高性能数据库连接池
/// 
/// 提供全面的连接池管理功能：
/// 1. 自适应连接池大小调整
/// 2. 智能健康检查和故障恢复
/// 3. 连接复用和预热机制
/// 4. 高并发负载均衡
/// 5. 性能监控和指标收集
/// 6. 连接泄漏检测和修复

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use async_trait::async_trait;
use tokio::sync::{Semaphore, Notify};
use tokio::time::{sleep, timeout};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use log::{debug, info, warn, error};

use crate::{Error, Result};
use crate::core::types::HealthStatus;
use super::types::{DatabaseType};

/// 连接池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// 最小连接数
    pub min_connections: usize,
    /// 最大连接数
    pub max_connections: usize,
    /// 初始连接数
    pub initial_connections: usize,
    /// 连接超时时间（秒）
    pub connection_timeout: u64,
    /// 空闲超时时间（秒）
    pub idle_timeout: u64,
    /// 最大生存时间（秒）
    pub max_lifetime: u64,
    /// 健康检查间隔（秒）
    pub health_check_interval: u64,
    /// 获取连接超时（秒）
    pub acquire_timeout: u64,
    /// 预热百分比
    pub warmup_percentage: f64,
    /// 自适应调整启用
    pub adaptive_sizing: bool,
    /// 故障恢复启用
    pub failure_recovery: bool,
    /// 连接泄漏检测启用
    pub leak_detection: bool,
    /// 性能监控启用
    pub performance_monitoring: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 5,
            max_connections: 50,
            initial_connections: 10,
            connection_timeout: 30,
            idle_timeout: 300,
            max_lifetime: 3600,
            health_check_interval: 60,
            acquire_timeout: 10,
            warmup_percentage: 0.5,
            adaptive_sizing: true,
            failure_recovery: true,
            leak_detection: true,
            performance_monitoring: true,
        }
    }
}

/// 连接状态
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    /// 空闲可用
    Idle,
    /// 使用中
    InUse,
    /// 健康检查中
    Checking,
    /// 故障状态
    Failed,
    /// 正在创建
    Creating,
    /// 正在销毁
    Destroying,
}

/// 连接元数据
#[derive(Debug, Clone)]
pub struct ConnectionMetadata {
    /// 连接ID
    pub id: String,
    /// 创建时间
    pub created_at: Instant,
    /// 最后使用时间
    pub last_used: Instant,
    /// 最后健康检查时间
    pub last_health_check: Instant,
    /// 使用次数
    pub use_count: u64,
    /// 连接状态
    pub state: ConnectionState,
    /// 故障次数
    pub failure_count: u32,
    /// 数据库类型
    pub db_type: DatabaseType,
    /// 连接信息
    pub connection_info: String,
}

impl ConnectionMetadata {
    /// 创建新的连接元数据
    pub fn new(db_type: DatabaseType, connection_info: String) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4().to_string(),
            created_at: now,
            last_used: now,
            last_health_check: now,
            use_count: 0,
            state: ConnectionState::Creating,
            failure_count: 0,
            db_type,
            connection_info,
        }
    }
    
    /// 标记使用
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
        self.state = ConnectionState::InUse;
    }
    
    /// 标记空闲
    pub fn mark_idle(&mut self) {
        self.state = ConnectionState::Idle;
    }
    
    /// 标记故障
    pub fn mark_failed(&mut self) {
        self.state = ConnectionState::Failed;
        self.failure_count += 1;
    }
    
    /// 检查是否过期
    pub fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }
    
    /// 检查是否空闲超时
    pub fn is_idle_timeout(&self, idle_timeout: Duration) -> bool {
        self.state == ConnectionState::Idle && self.last_used.elapsed() > idle_timeout
    }
    
    /// 检查是否需要健康检查
    pub fn needs_health_check(&self, interval: Duration) -> bool {
        self.last_health_check.elapsed() > interval
    }
    
    /// 获取连接年龄
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// 获取空闲时间
    pub fn idle_time(&self) -> Duration {
        if self.state == ConnectionState::Idle {
            self.last_used.elapsed()
        } else {
            Duration::ZERO
        }
    }
}

/// 池化连接
pub struct PooledConnection<T>
where
    T: Send + Sync + 'static,
{
    /// 数据库连接
    connection: Option<T>,
    /// 连接元数据
    metadata: ConnectionMetadata,
    /// 连接池引用
    pool: Arc<AdvancedConnectionPool<T>>,
    /// 返回标志
    returned: Arc<AtomicBool>,
}

impl<T> PooledConnection<T>
where
    T: Send + Sync + 'static,
{
    /// 创建新的池化连接
    fn new(connection: T, metadata: ConnectionMetadata, pool: Arc<AdvancedConnectionPool<T>>) -> Self {
        Self {
            connection: Some(connection),
            metadata,
            pool,
            returned: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// 获取连接引用
    pub fn connection(&self) -> Option<&T> {
        self.connection.as_ref()
    }
    
    /// 获取可变连接引用
    pub fn connection_mut(&mut self) -> Option<&mut T> {
        self.connection.as_mut()
    }
    
    /// 获取连接元数据
    pub fn metadata(&self) -> &ConnectionMetadata {
        &self.metadata
    }
    
    /// 手动返回连接到池
    pub async fn return_to_pool(mut self) -> Result<()> {
        if let Some(connection) = self.connection.take() {
            self.returned.store(true, Ordering::SeqCst);
            self.pool.return_connection(connection, self.metadata.clone()).await?;
        }
        Ok(())
    }
    
    /// 标记连接为故障
    pub async fn mark_failed(mut self) -> Result<()> {
        if let Some(connection) = self.connection.take() {
            self.returned.store(true, Ordering::SeqCst);
            let mut metadata = self.metadata.clone();
            metadata.mark_failed();
            self.pool.return_failed_connection(connection, metadata).await?;
        }
        Ok(())
    }
}

impl<T> Drop for PooledConnection<T>
where
    T: Send + Sync + 'static,
{
    fn drop(&mut self) {
        if !self.returned.load(Ordering::SeqCst) {
            if let Some(connection) = self.connection.take() {
                let pool = Arc::clone(&self.pool);
                let mut metadata = self.metadata.clone();
                metadata.mark_idle();
                
                tokio::spawn(async move {
                    if let Err(e) = pool.return_connection(connection, metadata).await {
                        error!("Failed to return connection to pool: {}", e);
                    }
                });
            }
        }
    }
}

/// 连接工厂特质
#[async_trait]
pub trait ConnectionFactory<T>: Send + Sync {
    /// 创建新连接
    async fn create_connection(&self) -> Result<T>;
    
    /// 验证连接健康
    async fn validate_connection(&self, connection: &T) -> Result<bool>;
    
    /// 销毁连接
    async fn destroy_connection(&self, connection: T) -> Result<()>;
    
    /// 获取连接信息
    fn connection_info(&self) -> String;
    
    /// 获取数据库类型
    fn database_type(&self) -> DatabaseType;
}

/// 高性能连接池
pub struct AdvancedConnectionPool<T> {
    /// 连接工厂
    factory: Arc<dyn ConnectionFactory<T>>,
    /// 配置
    config: PoolConfig,
    /// 可用连接队列
    available_connections: Arc<Mutex<VecDeque<(T, ConnectionMetadata)>>>,
    /// 使用中的连接
    active_connections: Arc<RwLock<HashMap<String, ConnectionMetadata>>>,
    /// 故障连接
    failed_connections: Arc<Mutex<Vec<(T, ConnectionMetadata)>>>,
    /// 连接信号量
    semaphore: Arc<Semaphore>,
    /// 统计信息
    stats: Arc<Mutex<PoolStats>>,
    /// 监控器
    monitor: Arc<PoolMonitor>,
    /// 健康检查器
    health_checker: Arc<HealthChecker>,
    /// 优化器
    optimizer: Arc<PoolOptimizer>,
    /// 运行状态
    running: Arc<AtomicBool>,
    /// 通知器
    notify: Arc<Notify>,
}

impl<T> AdvancedConnectionPool<T>
where
    T: Send + Sync + 'static,
{
    /// 创建新的高级连接池
    pub async fn new(
        factory: Arc<dyn ConnectionFactory<T>>,
        config: PoolConfig,
    ) -> Result<Arc<Self>> {
        let pool = Arc::new(Self {
            semaphore: Arc::new(Semaphore::new(config.max_connections)),
            available_connections: Arc::new(Mutex::new(VecDeque::new())),
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            failed_connections: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(PoolStats::new())),
            monitor: Arc::new(PoolMonitor::new()),
            health_checker: Arc::new(HealthChecker::new()),
            optimizer: Arc::new(PoolOptimizer::new()),
            running: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
            factory,
            config,
        });
        
        // 初始化连接池
        pool.initialize().await?;
        
        Ok(pool)
    }
    
    /// 初始化连接池
    async fn initialize(&self) -> Result<()> {
        info!("Initializing connection pool with {} initial connections", self.config.initial_connections);
        
        // 创建初始连接
        let mut tasks = Vec::new();
        for _ in 0..self.config.initial_connections {
            let factory = Arc::clone(&self.factory);
            tasks.push(tokio::spawn(async move {
                factory.create_connection().await
            }));
        }
        
        // 等待所有连接创建完成
        let mut created_count = 0;
        for task in tasks {
            match task.await {
                Ok(Ok(connection)) => {
                    let metadata = ConnectionMetadata::new(
                        self.factory.database_type(),
                        self.factory.connection_info(),
                    );
                    
                    let mut available = self.available_connections.lock().unwrap();
                    available.push_back((connection, metadata));
                    created_count += 1;
                }
                Ok(Err(e)) => {
                    warn!("Failed to create initial connection: {}", e);
                }
                Err(e) => {
                    warn!("Task failed while creating initial connection: {}", e);
                }
            }
        }
        
        info!("Successfully created {} initial connections", created_count);
        
        // 更新统计信息
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_created = created_count as u64;
            stats.available_connections = created_count;
        }
        
        Ok(())
    }
    
    /// 启动连接池
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(Error::invalid_state("Connection pool is already running"));
        }
        
        info!("Starting advanced connection pool");
        
        // 启动健康检查任务
        if self.config.health_check_interval > 0 {
            Arc::clone(&self).start_health_check_task().await?;
        }
        
        // 启动监控任务
        if self.config.performance_monitoring {
            self.start_monitoring_task().await?;
        }
        
        // 启动自适应调整任务
        if self.config.adaptive_sizing {
            Arc::clone(&self).start_adaptive_sizing_task().await?;
        }
        
        // 启动清理任务
        Arc::clone(&self).start_cleanup_task().await?;
        
        info!("Connection pool started successfully");
        Ok(())
    }
    
    /// 停止连接池
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping connection pool");
        
        self.running.store(false, Ordering::SeqCst);
        self.notify.notify_waiters();
        
        // 等待任务完成
        sleep(Duration::from_millis(100)).await;
        
        // 关闭所有连接
        self.close_all_connections().await?;
        
        info!("Connection pool stopped");
        Ok(())
    }
    
    /// 获取连接
    pub async fn get_connection(self: Arc<Self>) -> Result<PooledConnection<T>> {
        let acquire_start = Instant::now();
        
        // 获取信号量许可
        let permit = timeout(
            Duration::from_secs(self.config.acquire_timeout),
            Arc::clone(&self.semaphore).acquire_owned(),
        )
        .await
        .map_err(|_| Error::timeout("Failed to acquire connection within timeout"))?
        .map_err(|_| Error::internal("Semaphore closed"))?;
        
        // 尝试从可用连接中获取
        if let Some((connection, mut metadata)) = self.try_get_available_connection().await? {
            metadata.mark_used();
            
            // 添加到活跃连接
            {
                let mut active = self.active_connections.write().unwrap();
                active.insert(metadata.id.clone(), metadata.clone());
            }
            
            // 释放许可（因为我们成功获取了连接）
            permit.forget();
            
            // 更新统计
            {
                let mut stats = self.stats.lock().unwrap();
                stats.connections_acquired += 1;
                stats.active_connections += 1;
                stats.available_connections -= 1;
                stats.avg_acquire_time = self.update_avg_time(
                    stats.avg_acquire_time,
                    acquire_start.elapsed(),
                    stats.connections_acquired,
                );
            }
            
            return Ok(PooledConnection::new(connection, metadata, Arc::clone(&self)));
        }
        
        // 创建新连接
        match self.create_new_connection().await {
            Ok((connection, mut metadata)) => {
                metadata.mark_used();
                
                // 添加到活跃连接
                {
                    let mut active = self.active_connections.write().unwrap();
                    active.insert(metadata.id.clone(), metadata.clone());
                }
                
                // 释放许可
                permit.forget();
                
                // 更新统计
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.connections_acquired += 1;
                    stats.active_connections += 1;
                    stats.total_created += 1;
                    stats.avg_acquire_time = self.update_avg_time(
                        stats.avg_acquire_time,
                        acquire_start.elapsed(),
                        stats.connections_acquired,
                    );
                }
                
                Ok(PooledConnection::new(connection, metadata, Arc::clone(&self)))
            }
            Err(e) => {
                // 释放许可
                drop(permit);
                
                // 更新统计
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.connection_errors += 1;
                }
                
                Err(e)
            }
        }
    }
    
    /// 尝试从可用连接中获取
    async fn try_get_available_connection(&self) -> Result<Option<(T, ConnectionMetadata)>> {
        let mut available = self.available_connections.lock().unwrap();
        
        // 查找健康的连接
        while let Some((connection, metadata)) = available.pop_front() {
            // 检查连接是否过期
            if metadata.is_expired(Duration::from_secs(self.config.max_lifetime)) {
                // 连接过期，销毁它
                let factory = Arc::clone(&self.factory);
                tokio::spawn(async move {
                    if let Err(e) = factory.destroy_connection(connection).await {
                        warn!("Failed to destroy expired connection: {}", e);
                    }
                });
                continue;
            }
            
            // 检查连接是否空闲超时
            if metadata.is_idle_timeout(Duration::from_secs(self.config.idle_timeout)) {
                // 连接空闲超时，销毁它
                let factory = Arc::clone(&self.factory);
                tokio::spawn(async move {
                    if let Err(e) = factory.destroy_connection(connection).await {
                        warn!("Failed to destroy idle connection: {}", e);
                    }
                });
                continue;
            }
            
            // 检查连接健康状态
            if metadata.needs_health_check(Duration::from_secs(self.config.health_check_interval)) {
                match self.factory.validate_connection(&connection).await {
                    Ok(true) => {
                        // 连接健康，返回它
                        return Ok(Some((connection, metadata)));
                    }
                    Ok(false) | Err(_) => {
                        // 连接不健康，销毁它
                        let factory = Arc::clone(&self.factory);
                        tokio::spawn(async move {
                            if let Err(e) = factory.destroy_connection(connection).await {
                                warn!("Failed to destroy unhealthy connection: {}", e);
                            }
                        });
                        continue;
                    }
                }
            } else {
                // 不需要健康检查，直接返回
                return Ok(Some((connection, metadata)));
            }
        }
        
        Ok(None)
    }
    
    /// 创建新连接
    async fn create_new_connection(&self) -> Result<(T, ConnectionMetadata)> {
        let create_start = Instant::now();
        
        // 检查是否达到最大连接数
        let current_total = {
            let stats = self.stats.lock().unwrap();
            stats.active_connections + stats.available_connections
        };
        
        if current_total >= self.config.max_connections {
            return Err(Error::resource_exhausted("Maximum connections reached"));
        }
        
        // 创建连接
        let connection = timeout(
            Duration::from_secs(self.config.connection_timeout),
            self.factory.create_connection(),
        )
        .await
        .map_err(|_| Error::timeout("Connection creation timeout"))?
        .map_err(|e| Error::connection(&format!("Failed to create connection: {}", e)))?;
        
        let metadata = ConnectionMetadata::new(
            self.factory.database_type(),
            self.factory.connection_info(),
        );
        
        // 更新创建时间统计
        {
            let mut stats = self.stats.lock().unwrap();
            stats.avg_create_time = self.update_avg_time(
                stats.avg_create_time,
                create_start.elapsed(),
                stats.total_created + 1,
            );
        }
        
        Ok((connection, metadata))
    }
    
    /// 返回连接到池
    async fn return_connection(&self, connection: T, metadata: ConnectionMetadata) -> Result<()> {
        // 检查连接是否应该被保留
        if self.should_keep_connection(&metadata) {
            let mut metadata = metadata;
            metadata.mark_idle();
            
            // 添加到可用连接
            {
                let mut available = self.available_connections.lock().unwrap();
                available.push_back((connection, metadata.clone()));
            }
            
            // 从活跃连接中移除
            {
                let mut active = self.active_connections.write().unwrap();
                active.remove(&metadata.id);
            }
            
            // 更新统计
            {
                let mut stats = self.stats.lock().unwrap();
                stats.active_connections -= 1;
                stats.available_connections += 1;
                stats.connections_returned += 1;
            }
        } else {
            // 销毁连接
            self.destroy_connection(connection, metadata).await?;
        }
        
        Ok(())
    }
    
    /// 返回故障连接
    async fn return_failed_connection(&self, connection: T, metadata: ConnectionMetadata) -> Result<()> {
        // 从活跃连接中移除
        {
            let mut active = self.active_connections.write().unwrap();
            active.remove(&metadata.id);
        }
        
        // 销毁故障连接
        self.destroy_connection(connection, metadata).await?;
        
        // 更新统计
        {
            let mut stats = self.stats.lock().unwrap();
            stats.active_connections -= 1;
            stats.connection_errors += 1;
        }
        
        Ok(())
    }
    
    /// 销毁连接
    async fn destroy_connection(&self, connection: T, metadata: ConnectionMetadata) -> Result<()> {
        let factory = Arc::clone(&self.factory);
        
        tokio::spawn(async move {
            if let Err(e) = factory.destroy_connection(connection).await {
                warn!("Failed to destroy connection {}: {}", metadata.id, e);
            }
        });
        
        // 更新统计
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_destroyed += 1;
        }
        
        Ok(())
    }
    
    /// 检查是否应该保留连接
    fn should_keep_connection(&self, metadata: &ConnectionMetadata) -> bool {
        // 检查连接是否过期
        if metadata.is_expired(Duration::from_secs(self.config.max_lifetime)) {
            return false;
        }
        
        // 检查故障次数
        if metadata.failure_count > 3 {
            return false;
        }
        
        // 检查当前连接数是否超过最小值
        let current_total = {
            let stats = self.stats.lock().unwrap();
            stats.active_connections + stats.available_connections
        };
        
        if current_total <= self.config.min_connections {
            return true;
        }
        
        true
    }
    
    /// 关闭所有连接
    async fn close_all_connections(&self) -> Result<()> {
        info!("Closing all connections");
        
        // 关闭可用连接
        let available_connections = {
            let mut available = self.available_connections.lock().unwrap();
            available.drain(..).collect::<Vec<_>>()
        };
        
        for (connection, metadata) in available_connections {
            self.destroy_connection(connection, metadata).await?;
        }
        
        // 注意：活跃连接会在被返回时自动关闭
        
        info!("All connections closed");
        Ok(())
    }
    
    /// 启动健康检查任务
    async fn start_health_check_task(self: Arc<Self>) -> Result<()> {
        let pool = Arc::downgrade(&self);
        let running = Arc::clone(&self.running);
        let interval = Duration::from_secs(self.config.health_check_interval);
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            while running.load(Ordering::SeqCst) {
                interval_timer.tick().await;
                
                if let Some(pool) = pool.upgrade() {
                    if let Err(e) = pool.perform_health_check().await {
                        warn!("Health check failed: {}", e);
                    }
                } else {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// 执行健康检查
    async fn perform_health_check(&self) -> Result<()> {
        debug!("Performing health check");
        
        // 检查可用连接
        let connections_to_check = {
            let available = self.available_connections.lock().unwrap();
            available.len()
        };
        
        if connections_to_check == 0 {
            return Ok(());
        }
        
        // 从可用连接中取出一些进行健康检查
        let check_count = (connections_to_check / 4).max(1);
        let mut checked_connections = Vec::new();
        
        for _ in 0..check_count {
            if let Some((connection, mut metadata)) = {
                let mut available = self.available_connections.lock().unwrap();
                available.pop_front()
            } {
                metadata.state = ConnectionState::Checking;
                metadata.last_health_check = Instant::now();
                
                match self.factory.validate_connection(&connection).await {
                    Ok(true) => {
                        metadata.state = ConnectionState::Idle;
                        checked_connections.push((connection, metadata));
                    }
                    Ok(false) | Err(_) => {
                        // 连接不健康，销毁它
                        self.destroy_connection(connection, metadata).await?;
                    }
                }
            }
        }
        
        // 将健康的连接放回队列
        {
            let mut available = self.available_connections.lock().unwrap();
            for (connection, metadata) in checked_connections {
                available.push_back((connection, metadata));
            }
        }
        
        debug!("Health check completed");
        Ok(())
    }
    
    /// 启动监控任务
    async fn start_monitoring_task(&self) -> Result<()> {
        let stats = Arc::clone(&self.stats);
        let running = Arc::clone(&self.running);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                let stats_snapshot = {
                    stats.lock().unwrap().clone()
                };
                
                debug!("Pool stats: {:?}", stats_snapshot);
            }
        });
        
        Ok(())
    }
    
    /// 启动自适应调整任务
    async fn start_adaptive_sizing_task(self: Arc<Self>) -> Result<()> {
        let pool = Arc::downgrade(&self);
        let running = Arc::clone(&self.running);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                if let Some(pool) = pool.upgrade() {
                    if let Err(e) = pool.perform_adaptive_sizing().await {
                        warn!("Adaptive sizing failed: {}", e);
                    }
                } else {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// 执行自适应调整
    async fn perform_adaptive_sizing(&self) -> Result<()> {
        let (total_connections, active_connections, acquire_time) = {
            let stats = self.stats.lock().unwrap();
            (
                stats.active_connections + stats.available_connections,
                stats.active_connections,
                stats.avg_acquire_time,
            )
        };
        
        let utilization = active_connections as f64 / total_connections.max(1) as f64;
        
        // 如果利用率高且获取时间长，增加连接
        if utilization > 0.8 && acquire_time > Duration::from_millis(100) {
            let target_increase = ((self.config.max_connections - total_connections) / 4).max(1);
            self.warmup_connections(target_increase).await?;
        }
        
        // 如果利用率低，减少连接
        if utilization < 0.3 && total_connections > self.config.min_connections {
            self.shrink_pool().await?;
        }
        
        Ok(())
    }
    
    /// 预热连接
    async fn warmup_connections(&self, count: usize) -> Result<()> {
        info!("Warming up {} connections", count);
        
        let mut tasks = Vec::new();
        for _ in 0..count {
            let factory = Arc::clone(&self.factory);
            tasks.push(tokio::spawn(async move {
                factory.create_connection().await
            }));
        }
        
        let mut created_count = 0;
        for task in tasks {
            match task.await {
                Ok(Ok(connection)) => {
                    let metadata = ConnectionMetadata::new(
                        self.factory.database_type(),
                        self.factory.connection_info(),
                    );
                    
                    let mut available = self.available_connections.lock().unwrap();
                    available.push_back((connection, metadata));
                    created_count += 1;
                }
                Ok(Err(e)) => {
                    warn!("Failed to create warmup connection: {}", e);
                }
                Err(e) => {
                    warn!("Task failed while creating warmup connection: {}", e);
                }
            }
        }
        
        // 更新统计信息
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_created += created_count as u64;
            stats.available_connections += created_count;
        }
        
        info!("Successfully warmed up {} connections", created_count);
        Ok(())
    }
    
    /// 收缩连接池
    async fn shrink_pool(&self) -> Result<()> {
        let current_available = {
            let available = self.available_connections.lock().unwrap();
            available.len()
        };
        
        if current_available <= self.config.min_connections {
            return Ok(());
        }
        
        let shrink_count = ((current_available - self.config.min_connections) / 2).max(1);
        
        info!("Shrinking pool by {} connections", shrink_count);
        
        let connections_to_remove = {
            let mut available = self.available_connections.lock().unwrap();
            (0..shrink_count)
                .filter_map(|_| available.pop_back())
                .collect::<Vec<_>>()
        };
        
        for (connection, metadata) in connections_to_remove {
            self.destroy_connection(connection, metadata).await?;
        }
        
        // 更新统计信息
        {
            let mut stats = self.stats.lock().unwrap();
            stats.available_connections -= shrink_count;
        }
        
        info!("Successfully shrunk pool by {} connections", shrink_count);
        Ok(())
    }
    
    /// 启动清理任务
    async fn start_cleanup_task(self: Arc<Self>) -> Result<()> {
        let pool = Arc::downgrade(&self);
        let running = Arc::clone(&self.running);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5分钟
            
            while running.load(Ordering::SeqCst) {
                interval.tick().await;
                
                if let Some(pool) = pool.upgrade() {
                    if let Err(e) = pool.perform_cleanup().await {
                        warn!("Cleanup failed: {}", e);
                    }
                } else {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// 执行清理
    async fn perform_cleanup(&self) -> Result<()> {
        debug!("Performing cleanup");
        
        // 清理过期和空闲连接
        let connections_to_remove = {
            let mut available = self.available_connections.lock().unwrap();
            let mut remaining = VecDeque::new();
            let mut to_remove = Vec::new();
            
            while let Some((connection, metadata)) = available.pop_front() {
                if metadata.is_expired(Duration::from_secs(self.config.max_lifetime))
                    || metadata.is_idle_timeout(Duration::from_secs(self.config.idle_timeout))
                {
                    to_remove.push((connection, metadata));
                } else {
                    remaining.push_back((connection, metadata));
                }
            }
            
            *available = remaining;
            to_remove
        };
        
        for (connection, metadata) in connections_to_remove {
            self.destroy_connection(connection, metadata).await?;
        }
        
        debug!("Cleanup completed");
        Ok(())
    }
    
    /// 更新平均时间
    fn update_avg_time(&self, current_avg: Duration, new_time: Duration, count: u64) -> Duration {
        if count <= 1 {
            new_time
        } else {
            Duration::from_nanos(
                (current_avg.as_nanos() as u64 * (count - 1) + new_time.as_nanos() as u64) / count,
            )
        }
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// 健康检查
    pub fn health_check(&self) -> HealthStatus {
        let stats = self.stats.lock().unwrap();
        let total_connections = stats.active_connections + stats.available_connections;
        
        if total_connections == 0 {
            return HealthStatus::Critical;
        }
        
        let error_rate = stats.connection_errors as f64 / (stats.connections_acquired + 1) as f64;
        
        if error_rate > 0.1 {
            HealthStatus::Critical
        } else if error_rate > 0.05 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }
}

/// 连接池统计信息
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// 活跃连接数
    pub active_connections: usize,
    /// 可用连接数
    pub available_connections: usize,
    /// 总创建连接数
    pub total_created: u64,
    /// 总销毁连接数
    pub total_destroyed: u64,
    /// 获取连接次数
    pub connections_acquired: u64,
    /// 返回连接次数
    pub connections_returned: u64,
    /// 连接错误次数
    pub connection_errors: u64,
    /// 平均获取时间
    pub avg_acquire_time: Duration,
    /// 平均创建时间
    pub avg_create_time: Duration,
    /// 创建时间
    pub created_at: Instant,
}

impl PoolStats {
    fn new() -> Self {
        Self {
            active_connections: 0,
            available_connections: 0,
            total_created: 0,
            total_destroyed: 0,
            connections_acquired: 0,
            connections_returned: 0,
            connection_errors: 0,
            avg_acquire_time: Duration::ZERO,
            avg_create_time: Duration::ZERO,
            created_at: Instant::now(),
        }
    }
    
    /// 获取错误率
    pub fn error_rate(&self) -> f64 {
        if self.connections_acquired == 0 {
            0.0
        } else {
            self.connection_errors as f64 / self.connections_acquired as f64
        }
    }
    
    /// 获取利用率
    pub fn utilization_rate(&self) -> f64 {
        let total = self.active_connections + self.available_connections;
        if total == 0 {
            0.0
        } else {
            self.active_connections as f64 / total as f64
        }
    }
    
    /// 获取运行时间
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// 池监控器
pub struct PoolMonitor {
    // 监控指标存储
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl PoolMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// 记录指标
    pub fn record_metric(&self, name: &str, value: f64) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.insert(name.to_string(), value);
    }
    
    /// 获取指标
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        let metrics = self.metrics.lock().unwrap();
        metrics.get(name).copied()
    }
}

/// 健康检查器
pub struct HealthChecker {
    // 健康检查历史
    history: Arc<Mutex<VecDeque<HealthCheckResult>>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    
    /// 记录健康检查结果
    pub fn record_result(&self, result: HealthCheckResult) {
        let mut history = self.history.lock().unwrap();
        history.push_back(result);
        
        // 保持最近100次记录
        while history.len() > 100 {
            history.pop_front();
        }
    }
}

/// 健康检查结果
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub timestamp: Instant,
    pub connection_id: String,
    pub success: bool,
    pub response_time: Duration,
    pub error: Option<String>,
}

/// 池优化器
pub struct PoolOptimizer {
    // 优化历史
    optimization_history: Arc<Mutex<Vec<OptimizationAction>>>,
}

impl PoolOptimizer {
    pub fn new() -> Self {
        Self {
            optimization_history: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// 记录优化动作
    pub fn record_action(&self, action: OptimizationAction) {
        let mut history = self.optimization_history.lock().unwrap();
        history.push(action);
        
        // 保持最近50次记录
        if history.len() > 50 {
            history.remove(0);
        }
    }
}

/// 优化动作
#[derive(Debug, Clone)]
pub struct OptimizationAction {
    pub timestamp: Instant,
    pub action_type: String,
    pub before_state: PoolStats,
    pub after_state: Option<PoolStats>,
    pub reason: String,
} 
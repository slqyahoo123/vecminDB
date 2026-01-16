/// 并发工具模块
/// 
/// 提供高级并发原语、线程池管理、任务调度、
/// 分布式锁等功能，用于提升系统并发性能和可靠性

use std::sync::{Arc, Mutex, RwLock, Condvar};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use crossbeam_channel::{self, Sender, Receiver, TrySendError, RecvTimeoutError};
use tokio::sync::{Semaphore, broadcast, oneshot};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, warn, error, debug};

/// 任务优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// 任务状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// 任务定义
pub struct Task {
    pub id: Uuid,
    pub name: String,
    pub priority: TaskPriority,
    pub work: Box<dyn FnOnce() -> Result<(), String> + Send + 'static>,
    pub created_at: Instant,
    pub timeout: Option<Duration>,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("priority", &self.priority)
            .field("work", &"<closure>")
            .field("created_at", &self.created_at)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Task {
    pub fn new<F>(name: String, work: F) -> Self
    where
        F: FnOnce() -> Result<(), String> + Send + 'static,
    {
        Self {
            id: Uuid::new_v4(),
            name,
            priority: TaskPriority::default(),
            work: Box::new(work),
            created_at: Instant::now(),
            timeout: None,
        }
    }

    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// 任务结果
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: Uuid,
    pub task_name: String,
    pub status: TaskStatus,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub execution_time: Option<Duration>,
}

impl TaskResult {
    pub fn success(task_id: Uuid, task_name: String, started_at: Instant) -> Self {
        let completed_at = Instant::now();
        Self {
            task_id,
            task_name,
            status: TaskStatus::Completed,
            started_at: Some(started_at),
            completed_at: Some(completed_at),
            execution_time: Some(completed_at.duration_since(started_at)),
        }
    }

    pub fn failed(task_id: Uuid, task_name: String, error: String, started_at: Option<Instant>) -> Self {
        let completed_at = Instant::now();
        Self {
            task_id,
            task_name,
            status: TaskStatus::Failed(error),
            started_at,
            completed_at: Some(completed_at),
            execution_time: started_at.map(|s| completed_at.duration_since(s)),
        }
    }
}

/// 线程池配置
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub min_threads: usize,
    pub max_threads: usize,
    pub keep_alive: Duration,
    pub queue_capacity: usize,
    pub rejection_policy: RejectionPolicy,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            min_threads: 2,
            max_threads: num_cpus::get().max(4),
            keep_alive: Duration::from_secs(60),
            queue_capacity: 1000,
            rejection_policy: RejectionPolicy::Block,
        }
    }
}

/// 拒绝策略
#[derive(Debug, Clone, Copy)]
pub enum RejectionPolicy {
    Block,           // 阻塞直到有空间
    Reject,          // 立即拒绝
    DiscardOldest,   // 丢弃最旧的任务
}

/// 高级线程池
pub struct ThreadPool {
    config: ThreadPoolConfig,
    workers: Vec<Worker>,
    task_sender: Sender<Task>,
    task_receiver: Arc<Mutex<Receiver<Task>>>,
    result_sender: Sender<TaskResult>,
    pub result_receiver: Receiver<TaskResult>,
    shutdown: Arc<Mutex<bool>>,
    statistics: Arc<Mutex<PoolStatistics>>,
}

struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_cancelled: u64,
    pub active_threads: usize,
    pub total_threads: usize,
    pub queue_size: usize,
    pub average_execution_time: Duration,
}

impl ThreadPool {
    pub fn new(config: ThreadPoolConfig) -> Self {
        let (task_sender, task_receiver) = crossbeam_channel::unbounded();
        let (result_sender, result_receiver) = crossbeam_channel::unbounded();
        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let shutdown = Arc::new(Mutex::new(false));
        let statistics = Arc::new(Mutex::new(PoolStatistics::default()));

        let mut workers = Vec::with_capacity(config.max_threads);
        
        // 创建初始线程
        for id in 0..config.min_threads {
            workers.push(Worker::new(
                id,
                Arc::clone(&task_receiver),
                result_sender.clone(),
                Arc::clone(&shutdown),
                Arc::clone(&statistics),
            ));
        }

        Self {
            config,
            workers,
            task_sender,
            task_receiver,
            result_sender,
            result_receiver,
            shutdown,
            statistics,
        }
    }

    pub fn submit(&self, task: Task) -> Result<(), String> {
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.tasks_submitted += 1;
            stats.queue_size += 1;
        }

        match self.config.rejection_policy {
            RejectionPolicy::Block => {
                self.task_sender.send(task)
                    .map_err(|_| "线程池已关闭".to_string())?;
            }
            RejectionPolicy::Reject => {
                self.task_sender.try_send(task)
                    .map_err(|e| match e {
                        TrySendError::Full(_) => "任务队列已满".to_string(),
                        TrySendError::Disconnected(_) => "线程池已关闭".to_string(),
                    })?;
            }
            RejectionPolicy::DiscardOldest => {
                // 对于无界队列，DiscardOldest策略等同于Block
                // 如果需要真正的DiscardOldest行为，需要使用有界队列并手动管理
                self.task_sender.send(task)
                    .map_err(|_| "线程池已关闭".to_string())?;
            }
        }

        Ok(())
    }

    pub fn submit_with_callback<F, R>(&self, task: Task, callback: F) -> Result<(), String>
    where
        F: FnOnce(TaskResult) -> R + Send + 'static,
        R: Send + 'static,
    {
        // 这里可以扩展实现回调机制
        self.submit(task)
    }

    pub fn get_statistics(&self) -> PoolStatistics {
        (*self.statistics.lock().unwrap()).clone()
    }

    pub fn shutdown(self) {
        *self.shutdown.lock().unwrap() = true;
        drop(self.task_sender); // 关闭发送端

        for worker in self.workers {
            if let Some(thread) = worker.thread {
                thread.join().unwrap();
            }
        }
    }

    pub fn resize(&mut self, new_size: usize) -> Result<(), String> {
        if new_size < self.config.min_threads || new_size > self.config.max_threads {
            return Err("线程数超出配置范围".to_string());
        }

        let current_size = self.workers.len();
        if new_size > current_size {
            // 增加线程
            for id in current_size..new_size {
                self.workers.push(Worker::new(
                    id,
                    Arc::clone(&self.task_receiver),
                    self.result_sender.clone(),
                    Arc::clone(&self.shutdown),
                    Arc::clone(&self.statistics),
                ));
            }
        } else if new_size < current_size {
            // 减少线程（简化实现）
            // 实际实现需要优雅地停止多余的线程
            warn!("动态减少线程数量暂未实现");
        }

        Ok(())
    }
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<Receiver<Task>>>,
        result_sender: Sender<TaskResult>,
        shutdown: Arc<Mutex<bool>>,
        statistics: Arc<Mutex<PoolStatistics>>,
    ) -> Worker {
        let thread = thread::spawn(move || {
            info!("工作线程 {} 启动", id);
            
            loop {
                // 检查是否需要关闭
                if *shutdown.lock().unwrap() {
                    break;
                }

                // 尝试获取任务
                let task = {
                    let receiver = receiver.lock().unwrap();
                    receiver.recv_timeout(Duration::from_millis(100))
                };

                match task {
                    Ok(task) => {
                        let started_at = Instant::now();
                        debug!("工作线程 {} 开始执行任务 {}", id, task.name);

                        // 更新统计信息
                        {
                            let mut stats = statistics.lock().unwrap();
                            stats.queue_size = stats.queue_size.saturating_sub(1);
                        }

                        // 执行任务
                        let result = match task.timeout {
                            Some(timeout) => {
                                // 有超时限制的任务执行
                                let start = Instant::now();
                                let work_result = (task.work)();
                                
                                if start.elapsed() > timeout {
                                    TaskResult::failed(
                                        task.id,
                                        task.name,
                                        "任务执行超时".to_string(),
                                        Some(started_at),
                                    )
                                } else {
                                    match work_result {
                                        Ok(()) => TaskResult::success(task.id, task.name, started_at),
                                        Err(e) => TaskResult::failed(task.id, task.name, e, Some(started_at)),
                                    }
                                }
                            }
                            None => {
                                // 无超时限制的任务执行
                                match (task.work)() {
                                    Ok(()) => TaskResult::success(task.id, task.name, started_at),
                                    Err(e) => TaskResult::failed(task.id, task.name, e, Some(started_at)),
                                }
                            }
                        };

                        // 更新统计信息
                        {
                            let mut stats = statistics.lock().unwrap();
                            match &result.status {
                                TaskStatus::Completed => stats.tasks_completed += 1,
                                TaskStatus::Failed(_) => stats.tasks_failed += 1,
                                TaskStatus::Cancelled => stats.tasks_cancelled += 1,
                                _ => {}
                            }
                            
                            if let Some(exec_time) = result.execution_time {
                                // 简单的移动平均
                                let current_avg = stats.average_execution_time;
                                let total_completed = stats.tasks_completed + stats.tasks_failed;
                                if total_completed > 0 {
                                    let current_avg_ms = current_avg.as_millis();
                                    let exec_time_ms = exec_time.as_millis();
                                    let total_completed_u128 = total_completed as u128;
                                    stats.average_execution_time = 
                                        Duration::from_millis(
                                            ((current_avg_ms * (total_completed_u128 - 1) + exec_time_ms) / total_completed_u128) as u64
                                        );
                                }
                            }
                        }

                        // 发送结果
                        if let Err(_) = result_sender.send(result) {
                            warn!("无法发送任务结果");
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        // 超时，继续下一次循环
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        // 通道已断开，退出
                        break;
                    }
                }
            }

            info!("工作线程 {} 退出", id);
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }
}

/// 异步任务调度器
pub struct AsyncTaskScheduler {
    semaphore: Arc<Semaphore>,
    max_permits: usize,
    shutdown_sender: broadcast::Sender<()>,
    _shutdown_receiver: broadcast::Receiver<()>,
    statistics: Arc<Mutex<AsyncSchedulerStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct AsyncSchedulerStats {
    pub active_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub total_permits: usize,
    pub available_permits: usize,
}

impl AsyncTaskScheduler {
    pub fn new(max_concurrent_tasks: usize) -> Self {
        let (shutdown_sender, shutdown_receiver) = broadcast::channel(1);
        let max_permits = max_concurrent_tasks;
        
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            max_permits,
            shutdown_sender,
            _shutdown_receiver: shutdown_receiver,
            statistics: Arc::new(Mutex::new(AsyncSchedulerStats::default())),
        }
    }

    pub async fn submit<F, T>(&self, task: F) -> Result<T, String>
    where
        F: Future<Output = Result<T, String>> + Send + 'static,
        T: Send + 'static,
    {
        let permit = self.semaphore.acquire().await
            .map_err(|_| "调度器已关闭".to_string())?;

        {
            let mut stats = self.statistics.lock().unwrap();
            stats.active_tasks += 1;
        }

        let result = task.await;

        {
            let mut stats = self.statistics.lock().unwrap();
            stats.active_tasks -= 1;
            match &result {
                Ok(_) => stats.completed_tasks += 1,
                Err(_) => stats.failed_tasks += 1,
            }
        }

        drop(permit);
        result
    }

    pub async fn submit_with_timeout<F, T>(
        &self,
        task: F,
        timeout: Duration,
    ) -> Result<T, String>
    where
        F: Future<Output = Result<T, String>> + Send + 'static,
        T: Send + 'static,
    {
        tokio::time::timeout(timeout, self.submit(task))
            .await
            .map_err(|_| "任务执行超时".to_string())?
    }

    pub fn get_statistics(&self) -> AsyncSchedulerStats {
        let mut stats = (*self.statistics.lock().unwrap()).clone();
        stats.total_permits = self.max_permits;
        stats.available_permits = self.semaphore.available_permits();
        stats
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown_sender.send(());
    }
}

/// 分布式锁管理器
pub struct DistributedLockManager {
    locks: Arc<RwLock<HashMap<String, LockInfo>>>,
    cleanup_interval: Duration,
}

#[derive(Debug, Clone)]
pub struct LockInfo {
    pub holder: String,
    pub acquired_at: Instant,
    pub expires_at: Option<Instant>,
    pub lock_type: LockType,
}

#[derive(Debug, Clone)]
pub enum LockType {
    Exclusive,
    Shared,
}

impl DistributedLockManager {
    pub fn new() -> Self {
        Self {
            locks: Arc::new(RwLock::new(HashMap::new())),
            cleanup_interval: Duration::from_secs(60),
        }
    }

    pub fn acquire_lock(
        &self,
        resource: &str,
        holder: &str,
        lock_type: LockType,
        ttl: Option<Duration>,
    ) -> Result<bool, String> {
        let mut locks = self.locks.write().unwrap();
        
        let expires_at = ttl.map(|t| Instant::now() + t);
        
        match locks.get(resource) {
            Some(existing) => {
                // 检查锁是否过期
                if let Some(expires) = existing.expires_at {
                    if Instant::now() >= expires {
                        // 锁已过期，可以获取
                        locks.insert(resource.to_string(), LockInfo {
                            holder: holder.to_string(),
                            acquired_at: Instant::now(),
                            expires_at,
                            lock_type,
                        });
                        return Ok(true);
                    }
                }
                
                // 检查锁类型兼容性
                match (&existing.lock_type, &lock_type) {
                    (LockType::Shared, LockType::Shared) => Ok(true),
                    _ => Ok(false), // 独占锁或与独占锁冲突
                }
            }
            None => {
                // 资源未被锁定
                locks.insert(resource.to_string(), LockInfo {
                    holder: holder.to_string(),
                    acquired_at: Instant::now(),
                    expires_at,
                    lock_type,
                });
                Ok(true)
            }
        }
    }

    pub fn release_lock(&self, resource: &str, holder: &str) -> Result<bool, String> {
        let mut locks = self.locks.write().unwrap();
        
        match locks.get(resource) {
            Some(lock_info) => {
                if lock_info.holder == holder {
                    locks.remove(resource);
                    Ok(true)
                } else {
                    Ok(false) // 不是锁的持有者
                }
            }
            None => Ok(false), // 锁不存在
        }
    }

    pub fn is_locked(&self, resource: &str) -> bool {
        let locks = self.locks.read().unwrap();
        
        match locks.get(resource) {
            Some(lock_info) => {
                // 检查是否过期
                if let Some(expires) = lock_info.expires_at {
                    Instant::now() < expires
                } else {
                    true // 无过期时间的锁
                }
            }
            None => false,
        }
    }

    pub fn get_lock_info(&self, resource: &str) -> Option<LockInfo> {
        let locks = self.locks.read().unwrap();
        locks.get(resource).cloned()
    }

    pub fn cleanup_expired_locks(&self) -> usize {
        let mut locks = self.locks.write().unwrap();
        let now = Instant::now();
        let mut removed = 0;
        
        locks.retain(|_, lock_info| {
            if let Some(expires) = lock_info.expires_at {
                if now >= expires {
                    removed += 1;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        
        removed
    }

    pub fn start_cleanup_task(&self) -> JoinHandle<()> {
        let locks = Arc::clone(&self.locks);
        let interval = self.cleanup_interval;
        
        thread::spawn(move || {
            loop {
                thread::sleep(interval);
                
                let mut locks = locks.write().unwrap();
                let now = Instant::now();
                
                locks.retain(|_, lock_info| {
                    if let Some(expires) = lock_info.expires_at {
                        now < expires
                    } else {
                        true
                    }
                });
            }
        })
    }
}

/// 并发限制器
pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
    max_permits: usize,
}

impl ConcurrencyLimiter {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_permits: max_concurrent,
        }
    }

    pub async fn acquire(&self) -> Result<ConcurrencyPermit, String> {
        let permit = self.semaphore.acquire().await
            .map_err(|_| "并发限制器已关闭".to_string())?;
        
        Ok(ConcurrencyPermit { _permit: permit })
    }

    pub fn try_acquire(&self) -> Option<ConcurrencyPermit> {
        self.semaphore.try_acquire()
            .ok()
            .map(move |permit| ConcurrencyPermit { _permit: permit })
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn max_permits(&self) -> usize {
        self.max_permits
    }
}

pub struct ConcurrencyPermit<'a> {
    _permit: tokio::sync::SemaphorePermit<'a>,
}

/// 条件变量等待器，用于复杂的线程同步场景
pub struct ConditionalWaiter {
    lock: Arc<Mutex<bool>>,
    condvar: Arc<Condvar>,
    timeout: Option<Duration>,
}

impl ConditionalWaiter {
    pub fn new() -> Self {
        Self {
            lock: Arc::new(Mutex::new(false)),
            condvar: Arc::new(Condvar::new()),
            timeout: None,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 等待条件满足
    pub fn wait(&self) -> Result<(), String> {
        let mut started = self.lock.lock().map_err(|e| format!("锁获取失败: {}", e))?;
        
        if let Some(timeout) = self.timeout {
            let (guard, result) = self.condvar.wait_timeout(started, timeout)
                .map_err(|e| format!("条件等待失败: {}", e))?;
            started = guard;
            
            if result.timed_out() {
                return Err("等待超时".to_string());
            }
        } else {
            started = self.condvar.wait(started)
                .map_err(|e| format!("条件等待失败: {}", e))?;
        }
        
        info!("条件等待完成");
        Ok(())
    }

    /// 通知等待的线程
    pub fn notify(&self) -> Result<(), String> {
        let mut started = self.lock.lock().map_err(|e| format!("锁获取失败: {}", e))?;
        *started = true;
        self.condvar.notify_all();
        info!("条件通知已发送");
        Ok(())
    }
}

/// 高级任务队列，使用VecDeque实现
pub struct AdvancedTaskQueue {
    queue: Arc<Mutex<VecDeque<PriorityTask>>>,
    capacity: usize,
    not_full: Arc<Condvar>,
    not_empty: Arc<Condvar>,
}

#[derive(Debug)]
struct PriorityTask {
    task: Task,
    priority: TaskPriority,
    enqueued_at: Instant,
}

impl AdvancedTaskQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
            not_full: Arc::new(Condvar::new()),
            not_empty: Arc::new(Condvar::new()),
        }
    }

    /// 入队任务（阻塞直到有空间）
    pub fn enqueue(&self, task: Task) -> Result<(), String> {
        let mut queue = self.queue.lock().map_err(|e| format!("队列锁获取失败: {}", e))?;
        
        // 等待队列有空间
        while queue.len() >= self.capacity {
            queue = self.not_full.wait(queue).map_err(|e| format!("条件等待失败: {}", e))?;
        }
        
        let priority_task = PriorityTask {
            priority: task.priority,
            task,
            enqueued_at: Instant::now(),
        };
        
        // 按优先级插入
        let insert_pos = queue.iter().position(|t| t.priority < priority_task.priority)
            .unwrap_or(queue.len());
        queue.insert(insert_pos, priority_task);
        
        self.not_empty.notify_one();
        debug!("任务已入队，当前队列长度: {}", queue.len());
        Ok(())
    }

    /// 出队任务（阻塞直到有任务）
    pub fn dequeue(&self) -> Result<Task, String> {
        let mut queue = self.queue.lock().map_err(|e| format!("队列锁获取失败: {}", e))?;
        
        // 等待队列有任务
        while queue.is_empty() {
            queue = self.not_empty.wait(queue).map_err(|e| format!("条件等待失败: {}", e))?;
        }
        
        let priority_task = queue.pop_front().ok_or("队列为空")?;
        self.not_full.notify_one();
        
        debug!("任务已出队，当前队列长度: {}", queue.len());
        Ok(priority_task.task)
    }

    /// 尝试出队任务（非阻塞）
    pub fn try_dequeue(&self) -> Option<Task> {
        if let Ok(mut queue) = self.queue.lock() {
            if let Some(priority_task) = queue.pop_front() {
                self.not_full.notify_one();
                return Some(priority_task.task);
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 异步任务Future包装器，使用Pin和Context
pub struct AsyncTaskWrapper<T> {
    inner: Pin<Box<dyn Future<Output = Result<T, String>> + Send>>,
    waker: Option<Waker>,
    completed: bool,
}

impl<T> AsyncTaskWrapper<T> {
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = Result<T, String>> + Send + 'static,
    {
        Self {
            inner: Box::pin(future),
            waker: None,
            completed: false,
        }
    }

    /// 设置唤醒器
    pub fn set_waker(&mut self, waker: Waker) {
        self.waker = Some(waker);
    }

    /// 检查是否已完成
    pub fn is_completed(&self) -> bool {
        self.completed
    }
}

impl<T> Future for AsyncTaskWrapper<T> {
    type Output = Result<T, String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            return Poll::Ready(Err("任务已完成".to_string()));
        }

        match self.inner.as_mut().poll(cx) {
            Poll::Ready(result) => {
                self.completed = true;
                info!("异步任务执行完成");
                Poll::Ready(result)
            }
            Poll::Pending => {
                debug!("异步任务等待中");
                Poll::Pending
            }
        }
    }
}

/// 一次性结果通道管理器
pub struct OneshotResultManager<T> {
    sender: Option<oneshot::Sender<T>>,
    receiver: Option<oneshot::Receiver<T>>,
}

impl<T> OneshotResultManager<T> {
    pub fn new() -> Self {
        let (sender, receiver) = oneshot::channel();
        Self {
            sender: Some(sender),
            receiver: Some(receiver),
        }
    }

    /// 发送结果
    pub fn send(mut self, value: T) -> Result<(), T> {
        if let Some(sender) = self.sender.take() {
            sender.send(value)
        } else {
            error!("发送器已被消费");
            Err(value)
        }
    }

    /// 接收结果
    pub async fn receive(mut self) -> Result<T, String> {
        if let Some(receiver) = self.receiver.take() {
            receiver.await.map_err(|e| format!("接收失败: {}", e))
        } else {
            error!("接收器已被消费");
            Err("接收器已被消费".to_string())
        }
    }
}

/// 错误监控和报告系统
pub struct ErrorMonitor {
    error_count: Arc<Mutex<HashMap<String, u64>>>,
    critical_errors: Arc<Mutex<VecDeque<CriticalError>>>,
    max_critical_errors: usize,
}

#[derive(Debug, Clone)]
struct CriticalError {
    message: String,
    timestamp: Instant,
    context: String,
}

impl ErrorMonitor {
    pub fn new(max_critical_errors: usize) -> Self {
        Self {
            error_count: Arc::new(Mutex::new(HashMap::new())),
            critical_errors: Arc::new(Mutex::new(VecDeque::with_capacity(max_critical_errors))),
            max_critical_errors,
        }
    }

    /// 记录错误
    pub fn record_error(&self, error_type: &str, message: &str, context: &str) {
        error!("错误记录: {} - {} (上下文: {})", error_type, message, context);
        
        // 更新错误计数
        if let Ok(mut counts) = self.error_count.lock() {
            *counts.entry(error_type.to_string()).or_insert(0) += 1;
        }
        
        // 如果是关键错误，保存详细信息
        if error_type.contains("critical") || error_type.contains("fatal") {
            if let Ok(mut critical) = self.critical_errors.lock() {
                if critical.len() >= self.max_critical_errors {
                    critical.pop_front();
                }
                critical.push_back(CriticalError {
                    message: message.to_string(),
                    timestamp: Instant::now(),
                    context: context.to_string(),
                });
            }
        }
    }

    /// 获取错误统计
    pub fn get_error_stats(&self) -> HashMap<String, u64> {
        match self.error_count.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => {
                error!("获取错误统计失败：锁中毒");
                HashMap::new()
            }
        }
    }

    /// 获取关键错误
    pub fn get_critical_errors(&self) -> Vec<CriticalError> {
        match self.critical_errors.lock() {
            Ok(guard) => guard.iter().cloned().collect(),
            Err(_) => {
                error!("获取关键错误失败：锁中毒");
                Vec::new()
            }
        }
    }
}

/// 工具函数
pub mod utils {
    use super::*;

    /// 创建默认线程池
    pub fn create_default_thread_pool() -> ThreadPool {
        ThreadPool::new(ThreadPoolConfig::default())
    }

    /// 创建CPU密集型线程池
    pub fn create_cpu_intensive_pool() -> ThreadPool {
        let config = ThreadPoolConfig {
            min_threads: num_cpus::get(),
            max_threads: num_cpus::get(),
            keep_alive: Duration::from_secs(30),
            queue_capacity: 100,
            rejection_policy: RejectionPolicy::Block,
        };
        ThreadPool::new(config)
    }

    /// 创建IO密集型线程池
    pub fn create_io_intensive_pool() -> ThreadPool {
        let config = ThreadPoolConfig {
            min_threads: 4,
            max_threads: num_cpus::get() * 4,
            keep_alive: Duration::from_secs(120),
            queue_capacity: 1000,
            rejection_policy: RejectionPolicy::Block,
        };
        ThreadPool::new(config)
    }

    /// 并行执行多个任务
    pub async fn parallel_execute<T, F, Fut>(
        tasks: Vec<F>,
        max_concurrent: usize,
    ) -> Vec<Result<T, String>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, String>> + Send + 'static,
        T: Send + 'static,
    {
        let scheduler = AsyncTaskScheduler::new(max_concurrent);
        let mut results = Vec::with_capacity(tasks.len());
        
        for task in tasks {
            let result = scheduler.submit(task()).await;
            results.push(result);
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_basic() {
        let pool = ThreadPool::new(ThreadPoolConfig::default());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task = Task::new("test_task".to_string(), move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        pool.submit(task).unwrap();
        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        pool.shutdown();
    }

    #[tokio::test]
    async fn test_async_scheduler() {
        let scheduler = AsyncTaskScheduler::new(2);
        
        let result = scheduler.submit(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok("test".to_string())
        }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test");
    }

    #[test]
    fn test_distributed_lock() {
        let lock_manager = DistributedLockManager::new();
        
        // 获取独占锁
        assert!(lock_manager.acquire_lock("resource1", "holder1", LockType::Exclusive, None).unwrap());
        
        // 尝试获取相同资源的锁应该失败
        assert!(!lock_manager.acquire_lock("resource1", "holder2", LockType::Exclusive, None).unwrap());
        
        // 释放锁
        assert!(lock_manager.release_lock("resource1", "holder1").unwrap());
        
        // 现在应该可以获取锁了
        assert!(lock_manager.acquire_lock("resource1", "holder2", LockType::Exclusive, None).unwrap());
    }

    #[tokio::test]
    async fn test_concurrency_limiter() {
        let limiter = ConcurrencyLimiter::new(2);
        
        let permit1 = limiter.acquire().await.unwrap();
        let permit2 = limiter.acquire().await.unwrap();
        
        // 第三个应该无法立即获取
        assert!(limiter.try_acquire().is_none());
        
        drop(permit1);
        
        // 现在应该可以获取了
        assert!(limiter.try_acquire().is_some());
    }
} 
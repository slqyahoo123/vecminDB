// 向量并行工作线程池模块
//
// 提供工作线程池进行多线程任务处理

use std::thread;
use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicUsize;
use crossbeam_channel::{bounded, Sender, Receiver};

use crate::Result;
use crate::Error;

/// 工作任务
type Task = Box<dyn FnOnce() + Send + 'static>;

/// 工作线程池状态
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerPoolState {
    /// 未启动
    Inactive,
    /// 运行中
    Running,
    /// 正在关闭
    ShuttingDown,
    /// 已关闭
    Shutdown,
}

/// 工作线程池
pub struct WorkerPool {
    /// 工作线程数
    num_workers: usize,
    /// 最大队列长度
    max_queue_size: usize,
    /// 任务发送通道
    task_sender: Option<Sender<Task>>,
    /// 控制通道
    control_sender: Option<Sender<ControlMsg>>,
    /// 工作线程句柄
    workers: Vec<thread::JoinHandle<()>>,
    /// 状态
    state: WorkerPoolState,
}

/// 控制消息
enum ControlMsg {
    /// 停止工作线程池
    Shutdown,
}

impl WorkerPool {
    /// 创建新的工作线程池
    pub fn new(num_workers: usize, max_queue_size: usize) -> Self {
        Self {
            num_workers,
            max_queue_size,
            task_sender: None,
            control_sender: None,
            workers: Vec::new(),
            state: WorkerPoolState::Inactive,
        }
    }
    
    /// 启动工作线程池
    pub fn start(&mut self) -> Result<()> {
        if self.state != WorkerPoolState::Inactive {
            return Err(Error::vector("Worker pool is already running".to_string()));
        }
        
        // 创建任务通道
        let (task_tx, task_rx) = bounded(self.max_queue_size);
        
        // 创建控制通道
        let (control_tx, control_rx) = bounded(self.num_workers);
        
        // 启动工作线程
        let mut workers = Vec::with_capacity(self.num_workers);
        
        for id in 0..self.num_workers {
            let task_rx = task_rx.clone();
            let control_rx = control_rx.clone();
            
            let worker = thread::Builder::new()
                .name(format!("vector-worker-{}", id))
                .spawn(move || {
                    Worker::run(id, task_rx, control_rx);
                })
                .map_err(|e| Error::vector(format!("Failed to spawn worker thread: {}", e)))?;
                
            workers.push(worker);
        }
        
        self.task_sender = Some(task_tx);
        self.control_sender = Some(control_tx);
        self.workers = workers;
        self.state = WorkerPoolState::Running;
        
        Ok(())
    }
    
    /// 停止工作线程池
    pub fn stop(&mut self) -> Result<()> {
        if self.state != WorkerPoolState::Running {
            return Ok(());
        }
        
        self.state = WorkerPoolState::ShuttingDown;
        
        // 发送关闭消息给所有工作线程
        if let Some(control_sender) = &self.control_sender {
            for _ in 0..self.num_workers {
                control_sender.send(ControlMsg::Shutdown)
                    .map_err(|_| Error::vector("Failed to send shutdown signal to worker".to_string()))?;
            }
        }
        
        // 等待所有工作线程结束
        while let Some(worker) = self.workers.pop() {
            worker.join()
                .map_err(|_| Error::vector("Worker thread panicked during shutdown".to_string()))?;
        }
        
        // 清理资源
        self.task_sender = None;
        self.control_sender = None;
        self.state = WorkerPoolState::Shutdown;
        
        Ok(())
    }
    
    /// 提交任务到工作线程池
    pub fn submit<F>(&self, f: F) -> Result<()> 
    where 
        F: FnOnce() + Send + 'static,
    {
        if self.state != WorkerPoolState::Running {
            return Err(Error::vector("Worker pool is not running".to_string()));
        }
        
        if let Some(sender) = &self.task_sender {
            sender.send(Box::new(f))
                .map_err(|_| Error::vector("Failed to submit task to worker pool".to_string()))?;
            Ok(())
        } else {
            Err(Error::vector("Worker pool has no task channel".to_string()))
        }
    }
    
    /// 获取工作线程池状态
    pub fn state(&self) -> WorkerPoolState {
        self.state
    }
    
    /// 获取正在运行的工作线程数量
    pub fn num_workers(&self) -> usize {
        if self.state == WorkerPoolState::Running {
            self.workers.len()
        } else {
            0
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // 确保线程池被正确关闭
        let _ = self.stop();
    }
}

/// 工作线程
struct Worker;

impl Worker {
    /// 运行工作线程
    fn run(id: usize, task_rx: Receiver<Task>, control_rx: Receiver<ControlMsg>) {
        log::debug!("Worker {} started", id);
        
        loop {
            // 使用select选择任务或控制消息
            crossbeam_channel::select! {
                recv(task_rx) -> task_result => {
                    match task_result {
                        Ok(task) => {
                            // 执行任务
                            task();
                        },
                        Err(_) => {
                            // 任务通道关闭，退出线程
                            log::debug!("Worker {} task channel closed, exiting", id);
                            break;
                        }
                    }
                },
                recv(control_rx) -> control_result => {
                    match control_result {
                        Ok(ControlMsg::Shutdown) => {
                            // 收到关闭消息，退出线程
                            log::debug!("Worker {} received shutdown signal, exiting", id);
                            break;
                        },
                        Err(_) => {
                            // 控制通道关闭，退出线程
                            log::debug!("Worker {} control channel closed, exiting", id);
                            break;
                        }
                    }
                }
            }
        }
        
        log::debug!("Worker {} stopped", id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    
    #[test]
    fn test_worker_pool() {
        // 创建工作线程池
        let mut pool = WorkerPool::new(4, 10);
        pool.start().unwrap();
        
        // 创建共享计数器
        let counter = Arc::new(AtomicUsize::new(0));
        
        // 提交10个任务
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.submit(move || {
                // 模拟工作
                thread::sleep(Duration::from_millis(10));
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }).unwrap();
        }
        
        // 给任务一些时间执行
        thread::sleep(Duration::from_millis(200));
        
        // 停止工作线程池
        pool.stop().unwrap();
        
        // 验证所有任务都已执行
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }
}

/// 并发安全的工作线程池
pub struct ConcurrentWorkerPool {
    /// 工作线程数量
    worker_count: usize,
    /// 工作线程句柄
    workers: Vec<thread::JoinHandle<()>>,
    /// 任务发送器
    task_sender: Sender<Task>,
    /// 任务接收器
    task_receiver: Receiver<Task>,
    /// 线程池状态
    state: Arc<RwLock<WorkerPoolState>>,
    /// 任务计数器
    task_counter: Arc<AtomicUsize>,
    /// 完成计数器
    completed_counter: Arc<AtomicUsize>,
    /// 错误计数器
    error_counter: Arc<AtomicUsize>,
}

/// 工作线程池统计信息
#[derive(Debug, Clone)]
pub struct WorkerPoolStats {
    /// 总任务数
    pub total_tasks: usize,
    /// 已完成任务数
    pub completed_tasks: usize,
    /// 错误任务数
    pub error_tasks: usize,
    /// 活跃工作线程数
    pub active_workers: usize,
    /// 平均任务处理时间
    pub average_task_time: f64,
}

impl ConcurrentWorkerPool {
    /// 创建新的并发安全工作线程池
    pub fn new(worker_count: usize) -> Self {
        let (task_sender, task_receiver) = bounded(1000);
        let state = Arc::new(RwLock::new(WorkerPoolState::Inactive));
        let task_counter = Arc::new(AtomicUsize::new(0));
        let completed_counter = Arc::new(AtomicUsize::new(0));
        let error_counter = Arc::new(AtomicUsize::new(0));
        
        Self {
            worker_count,
            workers: Vec::new(),
            task_sender,
            task_receiver,
            state,
            task_counter,
            completed_counter,
            error_counter,
        }
    }
    
    /// 启动工作线程池
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.state.write().unwrap();
        if *state != WorkerPoolState::Inactive {
            return Err("工作线程池已在运行".into());
        }
        
        *state = WorkerPoolState::Running;
        drop(state);
        
        // 启动工作线程
        for i in 0..self.worker_count {
            let task_receiver = self.task_receiver.clone();
            let state = self.state.clone();
            let task_counter = self.task_counter.clone();
            let completed_counter = self.completed_counter.clone();
            let error_counter = self.error_counter.clone();
            
            let worker = thread::spawn(move || {
                Self::worker_loop(i, task_receiver, state, task_counter, completed_counter, error_counter);
            });
            
            self.workers.push(worker);
        }
        
        Ok(())
    }
    
    /// 停止工作线程池
    pub fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.state.write().unwrap();
        if *state != WorkerPoolState::Running {
            return Err("工作线程池未在运行".into());
        }
        
        *state = WorkerPoolState::ShuttingDown;
        drop(state);
        
        // 等待所有工作线程完成
        for worker in self.workers.drain(..) {
            worker.join().map_err(|_| "工作线程异常退出")?;
        }
        
        let mut state = self.state.write().unwrap();
        *state = WorkerPoolState::Shutdown;
        
        Ok(())
    }
    
    /// 提交任务
    pub fn submit_task(&self, task: Task) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.state.read().unwrap();
        if *state != WorkerPoolState::Running {
            return Err("工作线程池未在运行".into());
        }
        
        self.task_sender.send(task).map_err(|_| "任务提交失败")?;
        self.task_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(())
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> WorkerPoolStats {
        let total_tasks = self.task_counter.load(std::sync::atomic::Ordering::Relaxed);
        let completed_tasks = self.completed_counter.load(std::sync::atomic::Ordering::Relaxed);
        let error_tasks = self.error_counter.load(std::sync::atomic::Ordering::Relaxed);
        
        WorkerPoolStats {
            total_tasks,
            completed_tasks,
            error_tasks,
            active_workers: self.workers.len(),
            average_task_time: 0.0, // 这里可以实现更复杂的统计逻辑
        }
    }
    
    /// 工作线程主循环
    fn worker_loop(
        worker_id: usize,
        task_receiver: Receiver<Task>,
        state: Arc<RwLock<WorkerPoolState>>,
        task_counter: Arc<AtomicUsize>,
        completed_counter: Arc<AtomicUsize>,
        error_counter: Arc<AtomicUsize>,
    ) {
        loop {
            // 检查线程池状态
            let current_state = {
                let state = state.read().unwrap();
                *state
            };
            
            if current_state == WorkerPoolState::ShuttingDown || current_state == WorkerPoolState::Shutdown {
                break;
            }
            
            // 接收任务
            match task_receiver.recv() {
                Ok(task) => {
                    // 执行任务
                    let start_time = std::time::Instant::now();
                    
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        task();
                    })) {
                        Ok(_) => {
                            completed_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let duration = start_time.elapsed();
                            println!("工作线程 {} 完成任务，耗时: {:?}", worker_id, duration);
                        },
                        Err(_) => {
                            error_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            println!("工作线程 {} 任务执行失败", worker_id);
                        }
                    }
                },
                Err(_) => {
                    // 通道关闭，退出循环
                    break;
                }
            }
        }
        
        println!("工作线程 {} 退出", worker_id);
    }
    
    /// 重置计数器
    pub fn reset_counters(&self) {
        self.task_counter.store(0, std::sync::atomic::Ordering::Relaxed);
        self.completed_counter.store(0, std::sync::atomic::Ordering::Relaxed);
        self.error_counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// 获取任务计数
    pub fn get_task_count(&self) -> usize {
        self.task_counter.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 获取完成计数
    pub fn get_completed_count(&self) -> usize {
        self.completed_counter.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 获取错误计数
    pub fn get_error_count(&self) -> usize {
        self.error_counter.load(std::sync::atomic::Ordering::Relaxed)
    }
} 
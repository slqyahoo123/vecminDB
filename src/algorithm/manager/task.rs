// 任务管理模块
// 负责算法任务的创建、状态管理和结果处理

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc};
use log::{info, warn, error};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::algorithm::types::{AlgorithmTask, TaskStatus};
use super::config::AlgorithmManagerConfig;

/// 任务ID类型
pub type TaskId = String;

/// 任务执行结果
#[derive(Debug, Clone)]
pub struct TaskExecutionResult {
    pub task_id: TaskId,
    pub status: TaskStatus,
    pub result: Option<Value>,
    pub error: Option<String>,
    pub execution_time: f64,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub resource_usage: Option<ResourceUsage>,
}

/// 资源使用情况
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_bytes: usize,
    pub cpu_seconds: f64,
    pub network_bytes: Option<usize>,
    pub disk_bytes: Option<usize>,
    pub gpu_memory_bytes: Option<usize>,
}

/// 任务信息
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub id: TaskId,
    pub algorithm_id: String,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub parameters: HashMap<String, Value>,
    pub result: Option<Value>,
    pub error: Option<String>,
    /// 任务执行进度 (0.0-100.0)
    pub progress: f64,
}

impl From<AlgorithmTask> for TaskInfo {
    fn from(task: AlgorithmTask) -> Self {
        Self {
            id: task.id,
            algorithm_id: task.algorithm_id,
            status: task.status,
            created_at: task.created_at,
            updated_at: task.updated_at,
            completed_at: None, // 可以根据状态推断添加
            parameters: task.parameters,
            result: task.result,
            error: task.error.map(|e| e.message),
            progress: 0.0, // 默认进度为0
        }
    }
}

/// 任务管理器
pub struct TaskManager {
    /// 所有任务
    tasks: RwLock<HashMap<TaskId, AlgorithmTask>>,
    /// 运行中的任务
    running_tasks: Mutex<HashMap<TaskId, tokio::task::JoinHandle<Result<TaskExecutionResult>>>>,
    /// 配置
    config: AlgorithmManagerConfig,
}

impl TaskManager {
    /// 创建新的任务管理器
    pub fn new(config: AlgorithmManagerConfig) -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            running_tasks: Mutex::new(HashMap::new()),
            config,
        }
    }
    
    /// 创建新任务
    pub fn create_task(&self, algorithm_id: &str, parameters: HashMap<String, Value>) -> Result<TaskId> {
        let task = AlgorithmTask::new(algorithm_id, parameters);
        let task_id = task.id.clone();
        
        // 存储任务
        let mut tasks = self.tasks.write().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        tasks.insert(task_id.clone(), task);
        
        Ok(task_id)
    }
    
    /// 获取任务信息
    pub fn get_task(&self, task_id: &str) -> Result<Option<TaskInfo>> {
        let tasks = self.tasks.read().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        Ok(tasks.get(task_id).map(|task| TaskInfo::from(task.clone())))
    }
    
    /// 启动任务执行
    pub fn start_task<F>(&self, task_id: &str, executor: F) -> Result<()>
    where
        F: FnOnce(TaskId) -> Result<TaskExecutionResult> + Send + 'static
    {
        // 检查任务是否存在
        let mut tasks = self.tasks.write().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        let task = tasks.get_mut(task_id).ok_or_else(|| {
            Error::not_found(format!("任务不存在: {}", task_id))
        })?;
        
        // 检查并发限制
        let running_tasks = self.running_tasks.lock().map_err(|_| {
            Error::lock("无法获取运行任务锁")
        })?;
        
        if running_tasks.len() >= self.config.max_concurrent_tasks {
            return Err(Error::resource("超过最大并发任务数限制"));
        }
        
        // 更新任务状态
        task.status = TaskStatus::Running;
        task.updated_at = Utc::now();
        drop(tasks);
        
        // 创建异步任务
        let task_id_clone = task_id.to_string();
        let task_id_for_handle = task_id_clone.clone();
        let handle = tokio::spawn(async move {
            // 执行任务
            let result = executor(task_id_for_handle);
            
            // 处理任务结果
            // 注意：实际实现中，此处应该更新任务状态
            result
        });
        
        // 保存任务句柄
        let mut running_tasks = self.running_tasks.lock().map_err(|_| {
            Error::lock("无法获取运行任务锁")
        })?;
        
        running_tasks.insert(task_id_clone, handle);
        
        Ok(())
    }
    
    /// 等待任务完成或超时
    pub async fn wait_for_task(&self, task_id: &str, timeout: Option<Duration>) -> Result<TaskExecutionResult> {
        let timeout = timeout.unwrap_or(Duration::from_secs(self.config.default_task_timeout));
        
        // 获取任务句柄
        let handle = {
            let mut running_tasks = self.running_tasks.lock().map_err(|_| {
                Error::lock("无法获取运行任务锁")
            })?;
            
            running_tasks.remove(task_id).ok_or_else(|| {
                Error::not_found(format!("任务未在运行: {}", task_id))
            })?
        };
        
        // 等待任务完成或超时
        let result = tokio::time::timeout(timeout, handle).await;
        
        match result {
            Ok(Ok(result)) => {
                // 任务正常完成
                self.update_task_result(task_id, &result)?;
                Ok(result?)
            },
            Ok(Err(e)) => {
                // 任务运行出错
                error!("任务执行失败: {}, 错误: {}", task_id, e);
                
                // 更新任务状态为失败
                self.update_task_status(task_id, TaskStatus::Failed(format!("任务执行失败: {}", e)), None)?;
                
                Err(Error::internal(format!("任务执行失败: {}", e)))
            },
            Err(_) => {
                // 任务超时
                warn!("任务执行超时: {}", task_id);
                
                // 更新任务状态为失败
                self.update_task_status(
                    task_id, 
                    TaskStatus::Failed(format!("任务执行超时 ({}秒)", timeout.as_secs())), 
                    None
                )?;
                
                Err(Error::timeout(format!("任务执行超时: {}", task_id)))
            }
        }
    }
    
    /// 更新任务状态
    fn update_task_status(&self, task_id: &str, status: TaskStatus, error: Option<String>) -> Result<()> {
        let mut tasks = self.tasks.write().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        let task = tasks.get_mut(task_id).ok_or_else(|| {
            Error::not_found(format!("任务不存在: {}", task_id))
        })?;
        
        task.status = status;
        task.updated_at = Utc::now();
        
        if let Some(error_msg) = error {
            task.error_message = Some(error_msg);
        }
        
        Ok(())
    }
    
    /// 更新任务结果
    fn update_task_result(&self, task_id: &str, result: &Result<TaskExecutionResult>) -> Result<()> {
        let mut tasks = self.tasks.write().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        let task = tasks.get_mut(task_id).ok_or_else(|| {
            Error::not_found(format!("任务不存在: {}", task_id))
        })?;
        
        match result {
            Ok(execution_result) => {
                task.status = execution_result.status.clone();
                task.result = execution_result.result.clone();
                task.error_message = execution_result.error.clone();
            },
            Err(e) => {
                task.status = TaskStatus::Failed(format!("任务执行失败: {}", e));
                task.error_message = Some(format!("任务执行失败: {}", e));
            }
        }
        
        task.updated_at = Utc::now();
        
        Ok(())
    }
    
    /// 取消任务
    pub fn cancel_task(&self, task_id: &str) -> Result<()> {
        // 获取任务句柄并终止它
        let handle = {
            let mut running_tasks = self.running_tasks.lock().map_err(|_| {
                Error::lock("无法获取运行任务锁")
            })?;
            
            running_tasks.remove(task_id)
        };
        
        if let Some(handle) = handle {
            // 中断任务
            handle.abort();
            info!("已取消任务: {}", task_id);
        }
        
        // 更新任务状态
        self.update_task_status(task_id, TaskStatus::Cancelled, Some("任务已取消".to_string()))?;
        
        Ok(())
    }
    
    /// 清理过期任务
    pub fn cleanup_expired_tasks(&self) -> Result<usize> {
        let now = SystemTime::now();
        let ttl = Duration::from_secs(self.config.task_result_ttl);
        let expiry_time = now.checked_sub(ttl).unwrap_or(UNIX_EPOCH);
        
        let mut tasks = self.tasks.write().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        // 找出过期的已完成任务
        let expired_tasks: Vec<String> = tasks.iter()
            .filter(|(_, task)| {
                matches!(task.status, TaskStatus::Completed | TaskStatus::Failed(_) | TaskStatus::Cancelled) && 
                task.updated_at.timestamp() < expiry_time.duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
            })
            .map(|(id, _)| id.clone())
            .collect();
        
        // 删除过期任务
        for task_id in &expired_tasks {
            tasks.remove(task_id);
        }
        
        let count = expired_tasks.len();
        if count > 0 {
            info!("已清理 {} 个过期任务", count);
        }
        
        Ok(count)
    }
    
    /// 获取任务列表
    pub fn list_tasks(&self, status: Option<TaskStatus>, limit: Option<usize>) -> Result<Vec<TaskInfo>> {
        let tasks = self.tasks.read().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        let limit = limit.unwrap_or(100);
        
        let task_list: Vec<TaskInfo> = tasks.values()
            .filter(|task| {
                status.as_ref().map_or(true, |s| task.status == *s)
            })
            .take(limit)
            .map(|task| TaskInfo::from(task.clone()))
            .collect();
        
        Ok(task_list)
    }
} 
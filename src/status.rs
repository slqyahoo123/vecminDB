use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use tokio::sync::broadcast::{self, Sender, Receiver};
use crate::error::{Result, Error};
use thiserror::Error;
use async_trait::async_trait;
use log::{error, info, warn, debug};

/// 状态跟踪错误类型
#[derive(Error, Debug)]
pub enum StatusError {
    #[error("状态更新错误: {0}")]
    UpdateError(String),
    
    #[error("状态查询错误: {0}")]
    QueryError(String),
    
    #[error("状态不存在: {0}")]
    NotFound(String),
}

/// 状态类型定义
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatusType {
    /// 正在初始化
    Initializing,
    /// 等待中
    Waiting,
    /// 正在执行
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
    /// 数据准备中
    DataPreparing,
    /// 数据加载中
    DataLoading,
    /// 模型训练中
    ModelTraining,
    /// 算法应用中
    AlgorithmApplying,
    /// 结果验证中
    ResultValidating,
    /// 清理中
    Cleaning,
    /// 准备中
    Pending,
}

impl fmt::Display for StatusType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatusType::Initializing => write!(f, "初始化中"),
            StatusType::Waiting => write!(f, "等待中"),
            StatusType::Running => write!(f, "运行中"),
            StatusType::Paused => write!(f, "已暂停"),
            StatusType::Completed => write!(f, "已完成"),
            StatusType::Failed => write!(f, "失败"),
            StatusType::Cancelled => write!(f, "已取消"),
            StatusType::DataPreparing => write!(f, "数据准备中"),
            StatusType::DataLoading => write!(f, "数据加载中"),
            StatusType::ModelTraining => write!(f, "模型训练中"),
            StatusType::AlgorithmApplying => write!(f, "算法应用中"),
            StatusType::ResultValidating => write!(f, "结果验证中"),
            StatusType::Cleaning => write!(f, "清理中"),
            StatusType::Pending => write!(f, "准备中"),
        }
    }
}

/// 状态更新事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusEvent {
    /// 任务ID
    pub task_id: Uuid,
    /// 状态类型
    pub status: StatusType,
    /// 进度（0-100）
    pub progress: Option<u8>,
    /// 状态消息
    pub message: String,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 额外数据
    pub metadata: HashMap<String, String>,
}

impl StatusEvent {
    /// 创建新的状态事件
    pub fn new(task_id: Uuid, status: StatusType, message: impl Into<String>) -> Self {
        Self {
            task_id,
            status,
            progress: None,
            message: message.into(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// 设置进度
    pub fn with_progress(mut self, progress: u8) -> Self {
        self.progress = Some(progress.min(100));
        self
    }

    /// 设置消息
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// 状态跟踪器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusTrackerConfig {
    /// 状态事件缓存大小
    pub event_buffer_size: usize,
    /// 状态保留时间（秒）
    pub retention_time_secs: u64,
    /// 是否启用自动清理
    pub auto_cleanup: bool,
    /// 清理间隔（秒）
    pub cleanup_interval_secs: u64,
    /// 最大任务数量
    pub max_tasks: usize,
}

impl Default for StatusTrackerConfig {
    fn default() -> Self {
        Self {
            event_buffer_size: 1000,
            retention_time_secs: 3600, // 1小时
            auto_cleanup: true,
            cleanup_interval_secs: 300, // 5分钟
            max_tasks: 10000,
        }
    }
}

/// 状态跟踪器trait，定义状态跟踪的接口
#[async_trait]
pub trait StatusTrackerTrait: Send + Sync {
    /// 更新任务状态
    async fn update_status(&self, event: StatusEvent) -> Result<()>;
    
    /// 获取任务状态
    async fn get_status(&self, task_id: Uuid) -> Result<Option<StatusEvent>>;
    
    /// 获取所有任务状态
    async fn get_all_statuses(&self) -> Result<HashMap<Uuid, StatusEvent>>;
    
    /// 订阅状态变更
    async fn subscribe(&self) -> Result<Receiver<StatusEvent>>;
    
    /// 检查任务是否存在
    async fn has_task(&self, task_id: Uuid) -> Result<bool>;
    
    /// 记录错误
    async fn record_error(&self, error: String) -> Result<()>;
    
    /// 移除任务状态
    async fn remove_task(&self, task_id: Uuid) -> Result<()>;
}

/// 状态跟踪器实现，用于跟踪各种任务的状态
#[derive(Clone)]
pub struct StatusTracker {
    /// 任务状态映射
    statuses: Arc<Mutex<HashMap<Uuid, StatusEvent>>>,
    /// 状态变更发送器
    sender: Sender<StatusEvent>,
}

impl StatusTracker {
    /// 创建新的状态跟踪器
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        Self {
            statuses: Arc::new(Mutex::new(HashMap::new())),
            sender,
        }
    }

    /// 更新任务状态
    pub fn update_status(&self, event: StatusEvent) -> Result<()> {
        // 更新状态映射
        {
            let mut statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
            statuses.insert(event.task_id, event.clone());
        }

        // 发送状态更新事件
        let _ = self.sender.send(event);
        
        Ok(())
    }

    /// 获取任务状态
    pub fn get_status(&self, task_id: &Uuid) -> Result<Option<StatusEvent>> {
        let statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        Ok(statuses.get(task_id).cloned())
    }

    /// 获取所有任务状态
    pub fn get_all_statuses(&self) -> Result<Vec<StatusEvent>> {
        let statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        Ok(statuses.values().cloned().collect())
    }

    /// 订阅状态更新
    pub fn subscribe(&self) -> Receiver<StatusEvent> {
        self.sender.subscribe()
    }

    /// 创建新任务并记录初始状态
    pub fn create_task(&self, task_type: &str) -> Result<Uuid> {
        let task_id = Uuid::new_v4();
        let event = StatusEvent::new(task_id, StatusType::Initializing, format!("创建任务: {}", task_type))
            .with_metadata("task_type", task_type);
        
        info!("创建新任务: {} (ID: {})", task_type, task_id);
        self.update_status(event)?;
        
        Ok(task_id)
    }

    /// 更新任务进度
    pub fn update_progress(&self, task_id: Uuid, progress: u8, message: impl Into<String>) -> Result<()> {
        if let Some(mut event) = self.get_status(&task_id)? {
            event.progress = Some(progress.min(100));
            event.message = message.into();
            event.updated_at = Utc::now();
            
            debug!("任务 {} 进度更新: {}% - {}", task_id, progress, event.message);
            self.update_status(event)?;
        }
        
        Ok(())
    }

    /// 配置状态跟踪器
    pub async fn configure(&self) -> Result<()> {
        let statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        log::info!("配置状态跟踪器 - 当前跟踪任务数: {}", statuses.len());
        
        // 清理过期状态（超过24小时的已完成任务）
        let now = Utc::now();
        let expire_duration = chrono::Duration::hours(24);
        let expired_count = statuses.iter()
            .filter(|(_, event)| {
                event.status == StatusType::Completed && 
                now.signed_duration_since(event.updated_at) > expire_duration
            })
            .count();
            
        if expired_count > 0 {
            log::info!("发现 {} 个过期状态记录，将在下次清理时移除", expired_count);
        }
        
        log::info!("状态跟踪器配置完成");
        Ok(())
    }
    
    /// 更新任务状态
    pub fn update_task_status(&self, task_id: Uuid, status: StatusType, message: impl Into<String>) -> Result<()> {
        if let Some(mut event) = self.get_status(&task_id)? {
            event.status = status;
            event.message = message.into();
            event.updated_at = Utc::now();
            
            // 如果任务完成，设置进度为100%
            if status == StatusType::Completed {
                event.progress = Some(100);
            }
            
            self.update_status(event)?;
        }
        
        Ok(())
    }

    /// 标记任务为失败
    pub fn mark_task_failed(&self, task_id: Uuid, error_message: impl Into<String>) -> Result<()> {
        warn!("任务 {} 失败: {}", task_id, error_message.into());
        self.update_task_status(task_id, StatusType::Failed, error_message)
    }

    /// 标记任务为完成
    pub fn mark_task_completed(&self, task_id: Uuid, message: impl Into<String>) -> Result<()> {
        self.update_task_status(task_id, StatusType::Completed, message)
    }

    /// 标记任务为取消
    pub fn mark_task_cancelled(&self, task_id: Uuid, reason: impl Into<String>) -> Result<()> {
        self.update_task_status(task_id, StatusType::Cancelled, reason)
    }

    /// 标记任务运行中
    pub fn mark_running(&self, task_id: &Uuid, message: impl Into<String>) -> Result<()> {
        let event = StatusEvent::new(*task_id, StatusType::Running, message.into());
        self.update_status(event)
    }

    /// 标记任务准备中
    pub fn mark_pending(&self, task_id: &Uuid, message: impl Into<String>) -> Result<()> {
        let event = StatusEvent::new(*task_id, StatusType::Pending, message.into());
        self.update_status(event)
    }
}

impl Default for StatusTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_status_tracker() {
        let tracker = StatusTracker::new();
        
        // 创建任务
        let task_id = tracker.create_task("测试任务").unwrap();
        
        // 订阅状态更新
        let mut receiver = tracker.subscribe();
        
        // 更新状态
        tracker.update_progress(task_id, 50, "进行中").unwrap();
        
        // 接收状态更新
        let received = receiver.recv().await.unwrap();
        assert_eq!(received.task_id, task_id);
        assert_eq!(received.progress, Some(50));
        
        // 标记完成
        tracker.mark_task_completed(task_id, "任务完成").unwrap();
        
        // 验证状态
        let status = tracker.get_status(&task_id).unwrap().unwrap();
        assert_eq!(status.status, StatusType::Completed);
        assert_eq!(status.progress, Some(100));
    }
}

// 为StatusTracker实现StatusTrackerTrait trait
#[async_trait]
impl StatusTrackerTrait for StatusTracker {
    async fn update_status(&self, event: StatusEvent) -> Result<()> {
        // 更新状态映射
        {
            let mut statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
            statuses.insert(event.task_id, event.clone());
        }

        // 发送状态更新事件
        let _ = self.sender.send(event);
        
        Ok(())
    }

    async fn get_status(&self, task_id: Uuid) -> Result<Option<StatusEvent>> {
        let statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        Ok(statuses.get(&task_id).cloned())
    }

    async fn get_all_statuses(&self) -> Result<HashMap<Uuid, StatusEvent>> {
        let statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        Ok(statuses.clone())
    }

    async fn subscribe(&self) -> Result<Receiver<StatusEvent>> {
        Ok(self.sender.subscribe())
    }

    async fn has_task(&self, task_id: Uuid) -> Result<bool> {
        let statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        Ok(statuses.contains_key(&task_id))
    }

    async fn remove_task(&self, task_id: Uuid) -> Result<()> {
        let mut statuses = self.statuses.lock().map_err(|_| Error::lock("无法锁定状态映射"))?;
        statuses.remove(&task_id);
        Ok(())
    }

    async fn record_error(&self, error: String) -> Result<()> {
        let error_msg = format!("状态跟踪器记录错误: {}", error);
        error!("{}", error_msg);
        Ok(())
    }
}

/// 默认状态跟踪器实现
pub struct DefaultStatusTracker {
    inner: StatusTracker,
}

impl DefaultStatusTracker {
    pub fn new() -> Self {
        Self {
            inner: StatusTracker::new(),
        }
    }
}

#[async_trait]
impl StatusTrackerTrait for DefaultStatusTracker {
    async fn update_status(&self, event: StatusEvent) -> Result<()> {
        self.inner.update_status(event)
    }

    async fn get_status(&self, task_id: Uuid) -> Result<Option<StatusEvent>> {
        self.inner.get_status(&task_id)
    }

    async fn get_all_statuses(&self) -> Result<HashMap<Uuid, StatusEvent>> {
        // 明确调用 trait 方法，避免与直接方法冲突
        StatusTrackerTrait::get_all_statuses(&self.inner).await
    }

    async fn subscribe(&self) -> Result<Receiver<StatusEvent>> {
        // 明确调用 trait 方法，避免与直接方法冲突
        StatusTrackerTrait::subscribe(&self.inner).await
    }

    async fn has_task(&self, task_id: Uuid) -> Result<bool> {
        // 明确调用 trait 方法，避免与直接方法冲突
        StatusTrackerTrait::has_task(&self.inner, task_id).await
    }

    async fn remove_task(&self, task_id: Uuid) -> Result<()> {
        // 明确调用 trait 方法，避免与直接方法冲突
        StatusTrackerTrait::remove_task(&self.inner, task_id).await
    }
    
    async fn record_error(&self, error: String) -> Result<()> {
        let error_event = StatusEvent {
            task_id: uuid::Uuid::new_v4(),
            status: StatusType::Failed,
            progress: None,
            message: format!("Error: {}", error),
            updated_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        // 明确调用 trait 方法，避免与直接方法冲突
        StatusTrackerTrait::update_status(&self.inner, error_event).await
    }
} 
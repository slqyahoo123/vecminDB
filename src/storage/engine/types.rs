use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// 导出信息结构体
#[derive(Debug, Clone)]
pub struct ExportInfo {
    pub export_id: String,
    pub url: String,
    pub size_bytes: u64,
}

/// 部署信息结构体
#[derive(Debug, Clone)]
pub struct DeploymentInfo {
    pub deployment_id: String,
    pub api_url: String,
    pub version: String,
}

/// 模型版本信息结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersionInfo {
    pub version: String,
    pub created_at: String,
    pub is_current: bool,
    pub metrics: Option<serde_json::Value>,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// 分布式训练任务信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTaskInfo {
    /// 任务ID
    pub task_id: String,
    /// 模型ID
    pub model_id: String,
    /// 节点任务列表
    pub nodes: Vec<DistributedNodeTask>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 任务状态
    pub status: DistributedTaskStatus,
    /// 数据分区策略
    pub partition_strategy: DataPartitionStrategy,
}

impl std::fmt::Display for DistributedTaskInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistributedTaskInfo {{ task_id: {}, model_id: {}, nodes: {}, status: {:?} }}", 
               self.task_id, self.model_id, self.nodes.len(), self.status)
    }
}

/// 分布式节点任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedNodeTask {
    /// 节点ID
    pub node_id: String,
    /// 任务ID
    pub task_id: String,
    /// 节点排名
    pub rank: usize,
    /// 节点配置
    pub config: crate::training::config::TrainingConfig,
    /// 任务状态
    pub status: TaskStatus,
    /// 开始时间
    pub started_at: DateTime<Utc>,
}

/// 分布式任务状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedTaskStatus {
    /// 准备中
    Preparing,
    /// 运行中
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed { error: String },
    /// 已取消
    Cancelled,
}

/// 数据分区策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataPartitionStrategy {
    /// 水平分片
    Sharding,
    /// 随机分区
    Random,
    /// 按标签分区
    ByLabel,
    /// 自定义分区
    Custom,
}

/// 任务状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    /// 初始化
    Initialized,
    /// 运行中
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed { error: String },
    /// 已取消
    Cancelled,
    /// 等待中
    Pending,
}

/// 存储统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatistics {
    /// 对象总数
    pub total_objects: u64,
    /// 总大小（字节）
    pub total_size_bytes: u64,
    /// 读操作次数
    pub read_operations: u64,
    /// 写操作次数
    pub write_operations: u64,
    /// 缓存命中率
    pub cache_hit_rate: f64,
    /// 平均操作时间（毫秒）
    pub average_operation_time_ms: f64,
    /// 内存使用峰值（字节）
    pub peak_memory_usage_bytes: u64,
    /// 活跃连接数
    pub active_connections: u32,
    /// 最后备份时间
    pub last_backup_time: u64,
    /// 数据库健康评分
    pub database_health_score: f64,
}

/// 详细存储指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedStorageMetrics {
    /// 读取次数
    pub read_count: u64,
    /// 写入次数
    pub write_count: u64,
    /// 缓存命中率
    pub cache_hit_rate: f64,
    /// 平均操作时间（毫秒）
    pub average_operation_time_ms: f64,
    /// 内存使用峰值（字节）
    pub peak_memory_usage_bytes: u64,
    /// 活跃连接数
    pub active_connections: u32,
    /// 最后备份时间
    pub last_backup_time: u64,
    /// 数据库健康评分
    pub database_health_score: f64,
} 
use crate::Result;
use crate::compat::Model;
use crate::data::{DataBatch, DataSchema};
use crate::algorithm::Algorithm;
use std::collections::HashMap;
use crate::algorithm::{AlgorithmTask, TaskStatus};

// Stub types for health monitoring
#[derive(Debug, Clone)]
pub struct DiskSpaceInfo {
    pub total: u64,
    pub used: u64,
    pub available: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total: u64,
    pub used: u64,
    pub available: u64,
}

/// 存储隔离级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// 读未提交 - 可能读取到未提交的数据
    ReadUncommitted,
    /// 读已提交 - 只能读取已提交的数据
    ReadCommitted,
    /// 可重复读 - 同一事务内多次读取结果一致
    RepeatableRead,
    /// 串行化 - 最高隔离级别，完全串行执行
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        Self::ReadCommitted
    }
}

/// 事务状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionState {
    /// 活跃状态
    Active,
    /// 已提交
    Committed,
    /// 已回滚
    RolledBack,
    /// 部分失败
    PartiallyFailed,
}

/// 存储事务特征
pub trait StorageTransaction: Send + Sync {
    /// 获取事务ID
    fn id(&self) -> &str;
    
    /// 获取事务状态
    fn state(&self) -> TransactionState;
    
    /// 获取事务隔离级别
    fn isolation_level(&self) -> IsolationLevel;
    
    /// 提交事务
    fn commit(self: Box<Self>) -> Result<()>;
    
    /// 回滚事务
    fn rollback(self: Box<Self>) -> Result<()>;
    
    /// 向事务中添加数据
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// 从事务中获取数据
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// 从事务中删除数据
    fn delete(&mut self, key: &[u8]) -> Result<()>;
}

/// 存储接口
/// 
/// 定义与存储层交互的抽象接口，供db模块使用
pub trait StorageInterface: Send + Sync {
    /// 添加数据
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// 获取数据
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// 删除数据
    fn delete(&self, key: &[u8]) -> Result<()>;
    
    /// 检查键是否存在
    fn exists(&self, key: &[u8]) -> Result<bool>;
    
    /// 开始事务
    fn transaction(&self) -> Result<Box<dyn StorageTransaction>>;
    
    /// 使用指定隔离级别开始事务
    fn transaction_with_isolation(&self, isolation_level: IsolationLevel) -> Result<Box<dyn StorageTransaction>>;
    
    /// 获取活跃事务数量
    fn active_transaction_count(&self) -> Result<usize>;
    
    /// 清理过期事务
    async fn cleanup_stale_transactions(&self, older_than_seconds: u64) -> Result<usize>;
    
    // 模型相关操作
    
    /// 存储模型
    fn put_model(&self, model_id: &str, model: &Model) -> Result<()>;
    
    /// 获取模型
    fn get_model(&self, model_id: &str) -> Result<Model>;
    
    /// 删除模型
    fn delete_model(&self, model_id: &str) -> Result<()>;
    
    /// 列出所有模型
    fn list_models(&self) -> Result<Vec<String>>;
    
    // 数据相关操作
    
    /// 存储数据批次
    fn put_data_batch(&self, batch_id: &str, batch: &DataBatch) -> Result<()>;
    
    /// 获取数据批次
    fn get_data_batch(&self, batch_id: &str) -> Result<Option<DataBatch>>;
    
    /// 删除数据批次
    fn delete_data_batch(&self, batch_id: &str) -> Result<bool>;
    
    /// 列出所有数据批次
    fn list_data_batches(&self, prefix: Option<&str>) -> Result<Vec<String>>;
    
    /// 存储数据模式
    fn put_data_schema(&self, schema_id: &str, schema: &DataSchema) -> Result<()>;
    
    /// 获取数据模式
    fn get_data_schema(&self, schema_id: &str) -> Result<Option<DataSchema>>;
    
    /// 删除数据模式
    fn delete_data_schema(&self, schema_id: &str) -> Result<()>;
    
    /// 列出所有数据模式
    fn list_data_schemas(&self) -> Result<Vec<String>>;
    
    // 算法相关操作
    
    /// 存储算法
    fn put_algorithm(&self, algorithm_id: &str, algorithm: &Box<dyn Algorithm + Send + Sync>) -> Result<()>;
    
    /// 获取算法
    fn get_algorithm(&self, algorithm_id: &str) -> Result<Box<dyn Algorithm + Send + Sync>>;
    
    /// 删除算法
    fn delete_algorithm(&self, algorithm_id: &str) -> Result<()>;
    
    /// 列出所有算法
    fn list_algorithms(&self) -> Result<Vec<String>>;
    
    // 算法任务存储功能（生产级实现）
    
    /// 存储算法任务
    fn store_algorithm_task(&self, task: &AlgorithmTask) -> Result<()>;
    
    /// 获取算法任务
    fn get_algorithm_task(&self, task_id: &str) -> Result<Option<AlgorithmTask>>;
    
    /// 更新算法任务状态
    fn update_algorithm_task_status(&self, task_id: &str, status: TaskStatus) -> Result<()>;
    
    /// 设置算法任务结果
    fn set_algorithm_task_result(&self, task_id: &str, result: serde_json::Value) -> Result<()>;
    
    /// 设置算法任务错误
    fn set_algorithm_task_error(&self, task_id: &str, error: String) -> Result<()>;
    
    /// 删除算法任务
    fn delete_algorithm_task(&self, task_id: &str) -> Result<()>;
    
    /// 按状态查询算法任务
    fn get_tasks_by_status(&self, status: &TaskStatus) -> Result<Vec<AlgorithmTask>>;
    
    /// 按算法ID查询任务
    fn get_tasks_by_algorithm(&self, algorithm_id: &str) -> Result<Vec<AlgorithmTask>>;
    
    /// 获取所有待处理任务
    fn get_pending_tasks(&self) -> Result<Vec<AlgorithmTask>>;
    
    /// 获取正在运行的任务
    fn get_running_tasks(&self) -> Result<Vec<AlgorithmTask>>;
    
    /// 取消算法任务
    fn cancel_algorithm_task(&self, task_id: &str) -> Result<()>;
    
    /// 批量更新任务状态
    fn batch_update_task_status(&self, task_ids: &[String], status: TaskStatus) -> Result<()>;
    
    /// 清理已完成的任务（超过指定时间）
    fn cleanup_completed_tasks(&self, older_than_hours: u64) -> Result<usize>;
    
    /// 获取任务统计信息
    fn get_task_statistics(&self) -> Result<HashMap<String, u64>>;
    
    // 存储优化功能
    
    /// 优化存储性能配置
    fn optimize_storage_performance(&self) -> Result<()>;
    
    /// 存储健康检查
    fn storage_health_check(&self) -> Result<HashMap<String, String>>;
    
    /// 检查磁盘空间
    fn check_disk_space(&self) -> Result<DiskSpaceInfo>;
    
    /// 获取活跃连接数
    fn get_active_connections_count(&self) -> Result<usize>;
    
    /// 获取性能指标
    fn get_performance_metrics(&self) -> Result<crate::core::PerformanceMetrics>;
    
    /// 检查数据完整性
    fn check_data_integrity(&self) -> Result<bool>;
    
    /// 获取内存使用情况
    fn get_memory_usage(&self) -> Result<MemoryInfo>;
    
    // 元数据操作
    
    /// 更新批次元数据
    fn update_batch_metadata(&self, batch_id: &str, metadata: &HashMap<String, String>) -> Result<()>;
    
    /// 获取批次元数据
    fn get_batch_metadata(&self, batch_id: &str) -> Result<Option<HashMap<String, String>>>;
    
    /// 更新算法元数据
    fn update_algorithm_metadata(&self, algorithm_id: &str, metadata: &HashMap<String, String>) -> Result<()>;
    
    /// 获取算法元数据
    fn get_algorithm_metadata(&self, algorithm_id: &str) -> Result<Option<HashMap<String, String>>>;
    
    // 标签操作
    
    /// 给数据批次添加标签
    fn tag_data_batch(&self, batch_id: &str, tags: &[String]) -> Result<()>;
    
    /// 获取批次标签
    fn get_batch_tags(&self, batch_id: &str) -> Result<Vec<String>>;
    
    /// 根据标签获取批次
    fn get_batches_by_tag(&self, tag: &str) -> Result<Vec<String>>;
} 
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use crate::data::batch::DataBatch;
use crate::data::processor::types::ProcessorBatch;
use crate::data::processor::config::ProcessorConfig;
use crate::Error;

/// 数据处理器trait - 定义数据处理器的核心接口
#[async_trait]
pub trait DataProcessor: Send + Sync {
    /// 处理数据批次
    async fn process_batch(&self, batch: &DataBatch, config: &ProcessorConfig) -> Result<ProcessorBatch, Error>;
    
    /// 获取处理器名称
    fn name(&self) -> &str;
    
    /// 获取处理器版本
    fn version(&self) -> &str;
    
    /// 获取处理器配置
    fn config(&self) -> &ProcessorConfig;
    
    /// 验证输入数据
    async fn validate_input(&self, batch: &DataBatch) -> Result<bool, Error>;
    
    /// 获取处理器状态
    async fn get_status(&self) -> Result<ProcessorStatus, Error>;
    
    /// 重置处理器状态
    async fn reset(&self) -> Result<(), Error>;
    
    /// 获取处理器指标
    async fn get_metrics(&self) -> Result<HashMap<String, f64>, Error>;
    
    /// 检查是否支持特定格式
    fn supports_format(&self, format: &str) -> bool;
    
    /// 获取活跃任务数量
    async fn get_active_tasks_count(&self) -> Result<usize, Error>;
}

/// 批处理器trait - 支持批量数据处理
#[async_trait]
pub trait BatchProcessor: DataProcessor {
    /// 批量处理多个数据批次
    async fn process_batches(&self, batches: &[DataBatch], config: &ProcessorConfig) -> Result<Vec<ProcessorBatch>, Error>;
    
    /// 获取批处理配置
    fn batch_config(&self) -> &BatchProcessorConfig;
    
    /// 设置批处理大小
    fn set_batch_size(&mut self, size: usize);
    
    /// 获取最大并发数
    fn max_concurrency(&self) -> usize;
}

/// 流式处理器trait - 支持流式数据处理
#[async_trait]
pub trait StreamProcessor: DataProcessor {
    /// 处理数据流
    async fn process_stream<S>(&self, stream: S, config: &StreamProcessorConfig) -> Result<(), Error>
    where
        S: futures::Stream<Item = DataBatch> + Send + Unpin;
    
    /// 开始流式处理
    async fn start_streaming(&self, config: &StreamProcessorConfig) -> Result<(), Error>;
    
    /// 停止流式处理
    async fn stop_streaming(&self) -> Result<(), Error>;
    
    /// 获取流式处理状态
    async fn streaming_status(&self) -> Result<StreamingStatus, Error>;
}

/// 处理器状态枚举
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessorStatus {
    /// 空闲状态
    Idle,
    /// 处理中
    Processing,
    /// 已暂停
    Paused,
    /// 错误状态
    Error(String),
    /// 已停止
    Stopped,
}

/// 流式处理状态
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingStatus {
    /// 未启动
    NotStarted,
    /// 运行中
    Running,
    /// 已暂停
    Paused,
    /// 已停止
    Stopped,
    /// 错误状态
    Error(String),
}

/// 批处理器配置
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    /// 批大小
    pub batch_size: usize,
    /// 最大并发数
    pub max_concurrency: usize,
    /// 处理超时(秒)
    pub timeout_seconds: u64,
    /// 是否启用重试
    pub enable_retry: bool,
    /// 最大重试次数
    pub max_retries: usize,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_concurrency: 4,
            timeout_seconds: 300,
            enable_retry: true,
            max_retries: 3,
        }
    }
}

/// 流式处理器配置
#[derive(Debug, Clone)]
pub struct StreamProcessorConfig {
    /// 缓冲区大小
    pub buffer_size: usize,
    /// 处理间隔(毫秒)
    pub processing_interval_ms: u64,
    /// 背压策略
    pub backpressure_strategy: BackpressureStrategy,
    /// 是否启用检查点
    pub enable_checkpointing: bool,
    /// 检查点间隔
    pub checkpoint_interval_ms: u64,
}

impl Default for StreamProcessorConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            processing_interval_ms: 100,
            backpressure_strategy: BackpressureStrategy::Block,
            enable_checkpointing: false,
            checkpoint_interval_ms: 60000,
        }
    }
}

/// 背压策略
#[derive(Debug, Clone, PartialEq)]
pub enum BackpressureStrategy {
    /// 阻塞等待
    Block,
    /// 丢弃旧数据
    DropOldest,
    /// 丢弃新数据
    DropNewest,
    /// 缓存到磁盘
    Spillover,
}

/// 转换器trait - 用于数据转换操作
pub trait DataTransformer: Send + Sync {
    /// 转换数据
    fn transform(&self, input: &ProcessorBatch) -> Result<ProcessorBatch, Error>;
    
    /// 获取转换器名称
    fn name(&self) -> &str;
    
    /// 验证输入格式
    fn validate_input_format(&self, format: &str) -> bool;
    
    /// 获取输出格式
    fn output_format(&self) -> &str;
}

/// 验证器trait - 用于数据验证
pub trait DataValidator: Send + Sync {
    /// 验证数据批次
    fn validate(&self, batch: &DataBatch) -> Result<ValidationResult, Error>;
    
    /// 获取验证器名称
    fn name(&self) -> &str;
    
    /// 获取验证规则
    fn validation_rules(&self) -> &[ValidationRule];
}

use crate::core::interfaces::ValidationResult;

/// 验证统计
#[derive(Debug, Clone)]
pub struct ValidationStats {
    /// 总记录数
    pub total_records: usize,
    /// 有效记录数
    pub valid_records: usize,
    /// 无效记录数
    pub invalid_records: usize,
    /// 验证时间(毫秒)
    pub validation_time_ms: u64,
}

/// 验证规则
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// 字段名
    pub field_name: String,
    /// 规则类型
    pub rule_type: ValidationType,
    /// 规则参数
    pub parameters: HashMap<String, String>,
    /// 是否必需
    pub required: bool,
}

/// 验证类型
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationType {
    /// 数据类型验证
    DataType(String),
    /// 范围验证
    Range { min: f64, max: f64 },
    /// 长度验证
    Length { min: usize, max: usize },
    /// 正则表达式验证
    Regex(String),
    /// 枚举值验证
    Enum(Vec<String>),
    /// 自定义验证
    Custom(String),
}

/// 聚合器trait - 用于数据聚合操作
pub trait DataAggregator: Send + Sync {
    /// 聚合数据批次
    fn aggregate(&self, batches: &[ProcessorBatch]) -> Result<ProcessorBatch, Error>;
    
    /// 获取聚合器名称
    fn name(&self) -> &str;
    
    /// 获取聚合类型
    fn aggregation_type(&self) -> AggregationType;
}

/// 聚合类型
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationType {
    /// 求和
    Sum,
    /// 平均值
    Average,
    /// 计数
    Count,
    /// 最大值
    Max,
    /// 最小值
    Min,
    /// 分组聚合
    GroupBy(String),
    /// 自定义聚合
    Custom(String),
}

/// 处理器工厂trait
pub trait ProcessorFactory: Send + Sync {
    /// 创建数据处理器
    fn create_processor(&self, config: &ProcessorConfig) -> Result<Arc<dyn DataProcessor>, Error>;
    
    /// 获取支持的处理器类型
    fn supported_types(&self) -> Vec<String>;
    
    /// 验证配置
    fn validate_config(&self, config: &ProcessorConfig) -> Result<(), Error>;
}

/// 处理器注册表trait
pub trait ProcessorRegistry: Send + Sync {
    /// 注册处理器工厂
    fn register_factory(&mut self, processor_type: &str, factory: Arc<dyn ProcessorFactory>) -> Result<(), Error>;
    
    /// 获取处理器工厂
    fn get_factory(&self, processor_type: &str) -> Option<Arc<dyn ProcessorFactory>>;
    
    /// 列出所有注册的处理器类型
    fn list_types(&self) -> Vec<String>;
    
    /// 创建处理器实例
    fn create_processor(&self, processor_type: &str, config: &ProcessorConfig) -> Result<Arc<dyn DataProcessor>, Error>;
} 
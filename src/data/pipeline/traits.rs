// src/data/pipeline/traits.rs
//
// 数据导入流水线的共享接口
// 此模块定义了数据验证和流水线阶段的接口

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// 修复导入，使用存在的类型
use crate::data::record::Record;
use crate::data::{DataBatch, Dataset};
use crate::data::processor::ProcessedBatch;
use crate::data::processor::ProcessorConfig;
// 从自定义二进制格式导入RecordBatch
// CustomBinaryFormat is not referenced in current trait set
use crate::storage::engine::StorageEngine;
// 公开导出RecordBatch
pub use super::record_batch::RecordBatch;

/// 导入上下文
pub struct ImportContext {
    /// 源路径或标识符
    pub source: String,
    /// 处理器配置
    pub config: ProcessorConfig,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 开始时间
    pub start_time: Instant,
}

/// 导入结果
pub struct ImportResult {
    /// 导入ID
    pub id: String,
    /// 开始时间
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// 结束时间
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// 持续时间（毫秒）
    pub duration: Option<u64>,
    /// 是否成功
    pub success: bool,
    /// 结果消息
    pub message: String,
    /// 总记录数
    pub total_records: usize,
    /// 处理的记录数
    pub processed_records: usize,
    /// 失败的记录数
    pub records_failed: usize,
    /// 处理行数，与记录可能不同
    pub processed_rows: usize,
    /// 警告列表
    pub warnings: Vec<String>,
    /// 错误列表
    pub errors: Vec<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

/// 导出配置
pub struct ExportConfig {
    /// 是否包含表头
    pub include_header: bool,
    /// 字段分隔符
    pub delimiter: Option<String>,
    /// 字段列表
    pub fields: Option<Vec<String>>,
    /// 是否压缩
    pub compress: bool,
    /// 其他选项
    pub options: HashMap<String, String>,
}

/// 记录值类型
#[derive(Debug, Clone)]
pub enum RecordValue {
    /// 字符串值
    String(String),
    /// 数值
    Number(f64),
    /// 整数
    Integer(i64),
    /// 布尔值
    Boolean(bool),
    /// 空值
    Null,
    /// 数组
    Array(Vec<RecordValue>),
    /// 对象
    Object(HashMap<String, RecordValue>),
    /// 二进制数据
    Binary(Vec<u8>),
}

/// 为RecordValue实现Display特性，以便更好地格式化输出
impl fmt::Display for RecordValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordValue::String(s) => write!(f, "{}", s),
            RecordValue::Number(n) => write!(f, "{}", n),
            RecordValue::Integer(i) => write!(f, "{}", i),
            RecordValue::Boolean(b) => write!(f, "{}", b),
            RecordValue::Null => write!(f, "null"),
            RecordValue::Array(arr) => {
                write!(f, "[")?;
                let mut first = true;
                for value in arr {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", value)?;
                    first = false;
                }
                write!(f, "]")
            },
            RecordValue::Object(obj) => {
                write!(f, "{{")?;
                let mut first = true;
                for (key, value) in obj {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\": {}", key, value)?;
                    first = false;
                }
                write!(f, "}}")
            },
            RecordValue::Binary(data) => {
                if data.len() > 16 {
                    // 只显示前8个和后8个字节
                    write!(f, "<Binary: {:02X} {:02X} {:02X} {:02X} ... {:02X} {:02X} {:02X} {:02X}, {} bytes>", 
                          data[0], data[1], data[2], data[3], 
                          data[data.len()-4], data[data.len()-3], data[data.len()-2], data[data.len()-1],
                          data.len())
                } else {
                    write!(f, "<Binary: ")?;
                    for byte in data {
                        write!(f, "{:02X} ", byte)?;
                    }
                    write!(f, "{} bytes>", data.len())
                }
            },
        }
    }
}

/// 定义验证错误的处理策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationErrorStrategy {
    /// 遇到错误立即失败
    FailFast,
    /// 记录错误但继续处理
    CollectAndContinue,
    /// 跳过错误记录并继续处理
    SkipAndContinue,
}

/// 验证规则特征
pub trait ValidationRule: Send + Sync + 'static {
    /// 验证单个记录
    fn validate_record(&self, record: &Record) -> Result<(), Box<dyn Error>>;
    
    /// 验证批量记录
    fn validate_batch(&self, batch: &RecordBatch) -> Vec<Result<(), Box<dyn Error>>> {
        batch.records.iter().map(|record| self.validate_record(record)).collect()
    }
    
    /// 获取验证规则名称
    fn name(&self) -> &str;
    
    /// 获取验证规则描述
    fn description(&self) -> &str;
    
    /// 格式化验证结果
    fn format_result(&self, result: &Result<(), Box<dyn Error>>) -> String {
        match result {
            Ok(_) => format!("验证通过: {}", self.name()),
            Err(e) => format!("验证失败: {} - {}", self.name(), e),
        }
    }
    
    /// 获取错误处理策略
    fn error_strategy(&self) -> ValidationErrorStrategy {
        ValidationErrorStrategy::CollectAndContinue
    }
}

/// 流水线阶段状态
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineStageStatus {
    /// 未初始化
    NotInitialized,
    /// 初始化完成
    Initialized,
    /// 正在处理
    Processing,
    /// 处理完成
    Completed,
    /// 处理失败
    Failed,
    /// 已清理
    Cleaned,
}

/// 流水线阶段特征
pub trait PipelineStage: Send + Sync + 'static {
    /// 获取阶段名称
    fn name(&self) -> &str;
    
    /// 初始化阶段
    fn init(&mut self, context: &mut PipelineContext) -> Result<(), Box<dyn Error>>;
    
    /// 执行阶段处理
    fn process(&mut self, context: &mut PipelineContext) -> Result<(), Box<dyn Error>>;
    
    /// 清理阶段资源
    fn cleanup(&mut self, context: &mut PipelineContext) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    
    /// 获取阶段状态
    fn status(&self) -> PipelineStageStatus;
    
    /// 设置阶段状态
    fn set_status(&mut self, status: PipelineStageStatus);
    
    /// 获取阶段元数据
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// 日志记录处理结果
    fn log_result(&self, result: &Result<(), Box<dyn Error>>) {
        match result {
            Ok(_) => println!("阶段 [{}] 执行成功", self.name()),
            Err(e) => eprintln!("阶段 [{}] 执行失败: {}", self.name(), e),
        }
    }
    
    /// 获取依赖的阶段
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }
    
    /// 转换为Any类型（用于类型转换）
    fn as_any(&self) -> &dyn Any;
    
    /// 转换为可变Any类型
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// 流水线上下文，保存流水线执行的状态和数据
pub struct PipelineContext {
    /// 输入文件路径
    pub input_files: Vec<PathBuf>,
    
    /// 处理的记录
    pub records: Vec<Record>,
    
    /// 处理的记录批次
    pub batches: Vec<RecordBatch>,
    
    /// 流水线配置
    pub config: HashMap<String, String>,
    
    /// 共享数据（在阶段之间共享）
    pub shared_data: HashMap<String, Arc<Mutex<dyn Any + Send + Sync>>>,
    
    /// 存储引擎（如果有）
    pub storage_engine: Option<Arc<dyn StorageEngine>>,
    
    /// 元数据
    pub metadata: HashMap<String, String>,
    
    /// 错误记录
    pub errors: Vec<Box<dyn Error>>,
}

impl PipelineContext {
    /// 创建新的上下文
    pub fn new() -> Self {
        Self {
            input_files: Vec::new(),
            records: Vec::new(),
            batches: Vec::new(),
            config: HashMap::new(),
            shared_data: HashMap::new(),
            storage_engine: None,
            metadata: HashMap::new(),
            errors: Vec::new(),
        }
    }
    
    /// 添加共享数据
    pub fn add_shared_data<T: Any + Send + Sync>(&mut self, key: &str, data: T) {
        self.shared_data.insert(key.to_string(), Arc::new(Mutex::new(data)));
    }
    
    /// 获取共享数据
    pub fn get_shared_data<T: Any + Send + Sync>(&self, key: &str) -> Option<Arc<Mutex<T>>> {
        self.shared_data.get(key).and_then(|data| {
            let data_ref = data.clone();
            let lock = data_ref.lock().unwrap();
            if lock.type_id() == std::any::TypeId::of::<T>() {
                // 类型转换，这里是不安全的直接转换，实际使用时应当小心
                unsafe {
                    let ptr = Arc::into_raw(data.clone()) as *const Mutex<T>;
                    Some(Arc::from_raw(ptr))
                }
            } else {
                None
            }
        })
    }
}

/// 数据处理器特征
pub trait DataProcessor: Send + Sync + 'static {
    /// 处理单个记录
    fn process_record(&self, record: &mut Record) -> Result<(), Box<dyn Error>>;
    
    /// 处理批量记录
    fn process_batch(&self, batch: &mut RecordBatch) -> Result<(), Box<dyn Error>> {
        for record in &mut batch.records {
            self.process_record(record)?;
        }
        Ok(())
    }
    
    /// 获取处理器名称
    fn name(&self) -> &str;
    
    /// 初始化导入上下文
    fn initialize_import(&self, source: &str, config: &ProcessorConfig) -> Result<ImportContext, Box<dyn Error>>;
    
    /// 处理导入
    fn process_import(&self, context: ImportContext) -> Result<ImportResult, Box<dyn Error>>;
    
    /// 列出批次
    fn list_batches(&self, data_type: Option<&str>, status: Option<&str>, limit: usize) -> Result<Vec<DataBatch>, Box<dyn Error>>;
    
    /// 获取指定批次
    fn get_batch(&self, id: &str) -> Result<DataBatch, Box<dyn Error>>;
    
    /// 删除批次
    fn delete_batch(&self, id: &str) -> Result<(), Box<dyn Error>>;
    
    /// 获取批次样本
    fn get_batch_samples(&self, id: &str, limit: usize, offset: usize, fields: Option<&[String]>) -> Result<(DataBatch, Vec<HashMap<String, RecordValue>>), Box<dyn Error>>;
    
    /// 导出批次
    fn export_batch(&self, id: &str, format: &str, config: Option<&ExportConfig>) -> Result<(), Box<dyn Error>>;
    
    /// 列出数据集
    fn list_datasets(&self) -> Result<Vec<Dataset>, Box<dyn Error>>;
    
    /// 获取数据集
    fn get_dataset(&self, id: &str) -> Result<Dataset, Box<dyn Error>>;
    
    /// 删除数据集
    fn delete_dataset(&self, id: &str) -> Result<(), Box<dyn Error>>;
    
    /// 存储文件
    fn store_file(&self, id: &str, filename: &str, content_type: &str, data: Vec<u8>) -> Result<(), Box<dyn Error>>;
    
    /// 处理二进制数据
    fn process_binary_data(&self, batch_id: &str, data_type: &str, config: &ProcessorConfig, data: Vec<u8>) -> Result<ImportResult, Box<dyn Error>>;
    
    /// 更新数据集
    fn update_dataset(&self, dataset: &Dataset) -> Result<(), Box<dyn Error>>;
    
    /// 初始化文件导入
    fn initialize_file_import(&self, file_path: &str, config: &ProcessorConfig) -> Result<ImportContext, Box<dyn Error>>;
    
    /// 处理数据批次
    fn process_batch_with_config(&self, batch: &DataBatch, config: &ProcessorConfig) -> Result<ProcessedBatch, Box<dyn Error>>;
    
    /// 获取处理器状态
    fn get_status(&self) -> Result<serde_json::Value, Box<dyn Error>>;
    
    /// 获取活跃任务数量
    fn get_active_tasks_count(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64, Box<dyn Error>>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
    
    /// 导出数据
    fn export_data(&self, config: &ExportConfig) -> Result<Vec<RecordValue>, Box<dyn Error>>;
}

/// 流水线监控器特征
pub trait PipelineMonitor: Send + Sync + 'static {
    /// 开始监控阶段
    fn start_stage(&mut self, stage_name: &str);
    
    /// 结束监控阶段
    fn end_stage(&mut self, stage_name: &str, result: Result<(), Box<dyn Error>>);
    
    /// 记录指标
    fn record_metric(&mut self, key: &str, value: f64);
    
    /// 获取阶段时间
    fn get_stage_times(&self) -> HashMap<String, u128>;
    
    /// 获取指标
    fn get_metrics(&self) -> HashMap<String, Vec<f64>>;
}

/// 自定义错误类型，用于流水线处理
#[derive(Debug)]
pub struct PipelineError {
    message: String,
    stage: Option<String>,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl PipelineError {
    /// 创建一个新的流水线错误
    pub fn new<T: Into<String>>(message: T) -> Self {
        Self {
            message: message.into(),
            stage: None,
            source: None,
        }
    }
    
    /// 设置错误发生的阶段
    pub fn with_stage<T: Into<String>>(mut self, stage: T) -> Self {
        self.stage = Some(stage.into());
        self
    }
    
    /// 设置错误来源
    pub fn with_source<E: Error + Send + Sync + 'static>(mut self, source: E) -> Self {
        self.source = Some(Box::new(source));
        self
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(stage) = &self.stage {
            write!(f, "流水线错误 [{}]: {}", stage, self.message)?;
        } else {
            write!(f, "流水线错误: {}", self.message)?;
        }
        
        if let Some(source) = &self.source {
            write!(f, " - 原因: {}", source)?;
        }
        
        Ok(())
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Some(ref source) = self.source {
            Some(source.as_ref())
        } else {
            None
        }
    }
} 
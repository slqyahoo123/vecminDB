// Data Processor Core Types
// 数据处理器核心类型定义

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::time::Duration;
use serde_json::Value;
use std::collections::HashMap;
use crate::data::processor::types::Schema;

/// 处理器状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessorState {
    /// 初始状态
    Initial,
    /// 已准备
    Ready,
    /// 正在处理
    Processing,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
}

/// 处理器健康状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessorStatus {
    /// 初始状态/健康状态
    Healthy,
    /// 降级状态（部分功能受影响）
    Degraded,
    /// 不健康状态（严重问题）
    Unhealthy,
}

/// 处理器类型枚举 - 统一的处理器类型定义
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessorType {
    /// 文件处理器
    File,
    /// 数据库处理器
    Database,
    /// 流处理器
    Stream,
    /// HTTP处理器
    Http,
    /// 批处理器
    Batch,
    /// 实时处理器
    RealTime,
    /// 特征提取处理器
    FeatureExtraction,
    /// 归一化处理器
    Normalization,
    /// 自定义处理器
    Custom(String),
}

impl ProcessorType {
    /// 转换为字符串表示
    pub fn to_string(&self) -> String {
        match self {
            ProcessorType::File => "file".to_string(),
            ProcessorType::Database => "database".to_string(),
            ProcessorType::Stream => "stream".to_string(),
            ProcessorType::Http => "http".to_string(),
            ProcessorType::Batch => "batch".to_string(),
            ProcessorType::RealTime => "realtime".to_string(),
            ProcessorType::FeatureExtraction => "feature_extraction".to_string(),
            ProcessorType::Normalization => "normalization".to_string(),
            ProcessorType::Custom(name) => format!("custom:{}", name),
        }
    }

    /// 从字符串解析处理器类型
    pub fn from_string(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "file" => Some(ProcessorType::File),
            "database" => Some(ProcessorType::Database),
            "stream" => Some(ProcessorType::Stream),
            "http" => Some(ProcessorType::Http),
            "batch" => Some(ProcessorType::Batch),
            "realtime" => Some(ProcessorType::RealTime),
            "feature_extraction" => Some(ProcessorType::FeatureExtraction),
            "normalization" => Some(ProcessorType::Normalization),
            s if s.starts_with("custom:") => {
                Some(ProcessorType::Custom(s[7..].to_string()))
            },
            _ => None,
        }
    }

    /// 检查是否为自定义类型
    pub fn is_custom(&self) -> bool {
        matches!(self, ProcessorType::Custom(_))
    }

    /// 获取自定义类型名称
    pub fn custom_name(&self) -> Option<&str> {
        match self {
            ProcessorType::Custom(name) => Some(name),
            _ => None,
        }
    }
}

/// 任务句柄 - 表示正在执行的处理任务
#[derive(Debug, Clone)]
pub struct TaskHandle {
    pub id: String,
    pub status: ProcessorState,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub processor_type: ProcessorType,
    pub progress: f32, // 进度百分比 0.0-1.0
}

impl TaskHandle {
    /// 创建新的任务句柄
    pub fn new(id: String, processor_type: ProcessorType) -> Self {
        let now = Utc::now();
        Self {
            id,
            status: ProcessorState::Initial,
            created_at: now,
            updated_at: now,
            processor_type,
            progress: 0.0,
        }
    }

    /// 更新任务状态
    pub fn update_status(&mut self, status: ProcessorState) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// 更新任务进度
    pub fn update_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }

    /// 获取任务持续时间
    pub fn duration(&self) -> Duration {
        let now = Utc::now();
        (now - self.created_at).to_std().unwrap_or_default()
    }

    /// 检查任务是否完成
    pub fn is_completed(&self) -> bool {
        matches!(self.status, ProcessorState::Completed | ProcessorState::Failed | ProcessorState::Cancelled)
    }
}

/// 处理器性能指标
#[derive(Debug, Clone, Default)]
pub struct ProcessorMetrics {
    /// 处理的记录数
    pub processed_count: usize,
    /// 错误数量
    pub error_count: usize,
    /// 总处理时间
    pub total_processing_time: Duration,
    /// 平均处理时间
    pub average_processing_time: Duration,
    /// 最大处理时间
    pub max_processing_time: Duration,
    /// 最小处理时间
    pub min_processing_time: Duration,
    /// 成功率
    pub success_rate: f64,
    /// 吞吐量 (记录/秒)
    pub throughput: f64,
}

impl ProcessorMetrics {
    /// 创建新的指标
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录一次处理操作
    pub fn record_operation(&mut self, duration: Duration, success: bool) {
        self.processed_count += 1;
        if !success {
            self.error_count += 1;
        }
        
        self.total_processing_time += duration;
        self.average_processing_time = self.total_processing_time / self.processed_count as u32;
        
        if duration > self.max_processing_time {
            self.max_processing_time = duration;
        }
        
        if self.min_processing_time == Duration::ZERO || duration < self.min_processing_time {
            self.min_processing_time = duration;
        }
        
        self.success_rate = if self.processed_count > 0 {
            ((self.processed_count - self.error_count) as f64 / self.processed_count as f64) * 100.0
        } else {
            0.0
        };

        // 计算吞吐量 (记录/秒)
        if self.total_processing_time.as_secs() > 0 {
            self.throughput = self.processed_count as f64 / self.total_processing_time.as_secs_f64();
        }
    }

    /// 重置指标
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 合并其他指标
    pub fn merge(&mut self, other: &ProcessorMetrics) {
        self.processed_count += other.processed_count;
        self.error_count += other.error_count;
        self.total_processing_time += other.total_processing_time;
        
        if other.max_processing_time > self.max_processing_time {
            self.max_processing_time = other.max_processing_time;
        }
        
        if self.min_processing_time == Duration::ZERO || 
           (other.min_processing_time > Duration::ZERO && other.min_processing_time < self.min_processing_time) {
            self.min_processing_time = other.min_processing_time;
        }
        
        // 重新计算派生指标
        if self.processed_count > 0 {
            self.average_processing_time = self.total_processing_time / self.processed_count as u32;
            self.success_rate = ((self.processed_count - self.error_count) as f64 / self.processed_count as f64) * 100.0;
        }
        
        if self.total_processing_time.as_secs() > 0 {
            self.throughput = self.processed_count as f64 / self.total_processing_time.as_secs_f64();
        }
    }
}

/// 内存使用信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// 已使用字节数
    pub used_bytes: u64,
    /// 总字节数
    pub total_bytes: u64,
    /// 可用字节数
    pub available_bytes: u64,
    /// 使用率百分比
    pub usage_percentage: f64,
}

impl MemoryInfo {
    /// 创建新的内存信息
    pub fn new(used_bytes: u64, total_bytes: u64) -> Self {
        let available_bytes = total_bytes.saturating_sub(used_bytes);
        let usage_percentage = if total_bytes > 0 {
            (used_bytes as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };
        
        Self {
            used_bytes,
            total_bytes,
            available_bytes,
            usage_percentage,
        }
    }

    /// 检查是否超过阈值
    pub fn exceeds_threshold(&self, threshold_percentage: f64) -> bool {
        self.usage_percentage > threshold_percentage
    }

    /// 格式化为人类可读的字符串
    pub fn to_human_readable(&self) -> String {
        format!(
            "Used: {} / Total: {} ({:.1}%)",
            format_bytes(self.used_bytes),
            format_bytes(self.total_bytes),
            self.usage_percentage
        )
    }
}

/// 格式化字节数为人类可读形式
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}

/// 处理器配置结构
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub id: String,
    pub name: String,
    pub processor_type: ProcessorType,
    pub settings: HashMap<String, Value>,
    pub input_schema: Option<Schema>,
    pub output_schema: Option<Schema>,
    pub enabled: bool,
    pub priority: i32,
    pub max_retry_count: u32,
    pub timeout_seconds: u64,
}

impl ProcessorConfig {
    /// 创建新的处理器配置
    pub fn new(id: String, name: String, processor_type: ProcessorType) -> Self {
        Self {
            id,
            name,
            processor_type,
            settings: HashMap::new(),
            input_schema: None,
            output_schema: None,
            enabled: true,
            priority: 0,
            max_retry_count: 3,
            timeout_seconds: 300, // 5分钟默认超时
        }
    }

    /// 设置配置项
    pub fn set_setting(&mut self, key: String, value: Value) {
        self.settings.insert(key, value);
    }

    /// 获取配置项
    pub fn get_setting(&self, key: &str) -> Option<&Value> {
        self.settings.get(key)
    }

    /// 获取字符串配置项
    pub fn get_string_setting(&self, key: &str) -> Option<String> {
        self.settings.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    /// 获取布尔配置项
    pub fn get_bool_setting(&self, key: &str) -> Option<bool> {
        self.settings.get(key).and_then(|v| v.as_bool())
    }

    /// 获取数值配置项
    pub fn get_number_setting(&self, key: &str) -> Option<f64> {
        self.settings.get(key).and_then(|v| v.as_f64())
    }

    /// 验证配置有效性
    pub fn validate(&self) -> crate::Result<()> {
        if self.id.is_empty() {
            return Err(crate::Error::invalid_input("处理器ID不能为空"));
        }
        
        if self.name.is_empty() {
            return Err(crate::Error::invalid_input("处理器名称不能为空"));
        }
        
        if self.timeout_seconds == 0 {
            return Err(crate::Error::invalid_input("超时时间必须大于0"));
        }
        
        Ok(())
    }
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self::new(
            "default".to_string(),
            "Default Processor".to_string(),
            ProcessorType::Custom("default".to_string()),
        )
    }
}

/// 存储健康检查trait
#[async_trait::async_trait]
pub trait StorageHealthCheck: Send + Sync {
    /// 检查存储健康状态
    async fn health_check(&self) -> crate::Result<bool>;
    
    /// 获取详细健康信息
    async fn detailed_health_check(&self) -> crate::Result<HashMap<String, Value>>;
    
    /// 检查连接状态
    async fn check_connection(&self) -> crate::Result<bool>;
    
    /// 获取存储统计信息
    async fn get_storage_stats(&self) -> crate::Result<HashMap<String, u64>>;
}

/// 处理器trait - 定义处理器的基本接口
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// 获取处理器ID
    fn id(&self) -> &str;
    
    /// 获取处理器名称
    fn name(&self) -> &str;
    
    /// 获取处理器类型
    fn processor_type(&self) -> ProcessorType;
    
    /// 获取处理器配置
    fn config(&self) -> &ProcessorConfig;
    
    /// 获取处理器指标
    fn metrics(&self) -> &ProcessorMetrics;
    
    /// 检查处理器是否启用
    fn is_enabled(&self) -> bool {
        self.config().enabled
    }
    
    /// 启动处理器
    async fn start(&mut self) -> crate::Result<()>;
    
    /// 停止处理器
    async fn stop(&mut self) -> crate::Result<()>;
    
    /// 重启处理器
    async fn restart(&mut self) -> crate::Result<()> {
        self.stop().await?;
        self.start().await
    }
    
    /// 健康检查
    async fn health_check(&self) -> crate::Result<ProcessorStatus>;
    
    /// 处理数据
    async fn process(&mut self, data: Vec<u8>) -> crate::Result<Vec<u8>>;
    
    /// 获取支持的数据格式
    fn supported_formats(&self) -> Vec<crate::data::DataFormat>;
    
    /// 验证输入数据
    async fn validate_input(&self, data: &[u8]) -> crate::Result<bool>;
    
    /// 获取处理进度
    fn get_progress(&self) -> f32;
    
    /// 设置配置
    async fn update_config(&mut self, config: ProcessorConfig) -> crate::Result<()>;
} 
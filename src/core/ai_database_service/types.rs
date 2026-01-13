/// AI数据库服务类型定义模块
/// 
/// 包含请求、响应、配置等相关类型定义

use serde::{Serialize, Deserialize};
use crate::compat::data_to_model_engine::DataToModelConversionResult;
use crate::compat::end_to_end_pipeline::EndToEndTrainingResult;
use crate::compat::end_to_end_pipeline::DataSource;
use crate::compat::Model;
use crate::compat::data_to_model_engine::ConversionConfig;

/// 数据即AI模型请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataToAIModelRequest {
    /// 请求ID
    pub request_id: String,
    /// 数据源
    pub data_source: DataSource,
    /// 目标描述
    pub objective_description: String,
    /// 转换选项
    pub conversion_options: Option<ConversionConfig>,
    /// 训练选项
    pub training_options: Option<TrainingOptions>,
    /// 部署选项
    pub deployment_options: Option<DeploymentOptions>,
}

/// 训练选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingOptions {
    /// 是否自动训练
    pub auto_train: bool,
    /// 训练预算（秒）
    pub training_budget_seconds: Option<u64>,
    /// 质量目标
    pub quality_target: Option<f64>,
    /// 训练策略
    pub training_strategy: Option<String>,
}

/// 部署选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentOptions {
    /// 是否自动部署
    pub auto_deploy: bool,
    /// 部署环境
    pub deployment_environment: Option<String>,
    /// 扩展配置
    pub scaling_config: Option<ScalingConfig>,
}

/// 扩展配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// 最小实例数
    pub min_instances: u32,
    /// 最大实例数
    pub max_instances: u32,
    /// 自动扩展阈值
    pub auto_scaling_threshold: f64,
}

/// 数据即AI模型响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataToAIModelResponse {
    /// 请求ID
    pub request_id: String,
    /// 生成的模型
    pub generated_model: Model,
    /// 转换结果
    pub conversion_result: DataToModelConversionResult,
    /// 训练结果
    pub training_result: Option<EndToEndTrainingResult>,
    /// 部署信息
    pub deployment_info: Option<DeploymentInfo>,
    /// 性能指标
    pub performance_metrics: AIServicePerformanceMetrics,
    /// 处理时间统计
    pub processing_time_stats: ProcessingTimeStats,
}

/// 部署信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    /// 部署ID
    pub deployment_id: String,
    /// 服务端点
    pub service_endpoint: String,
    /// 部署状态
    pub deployment_status: String,
    /// 服务配置
    pub service_config: ServiceConfig,
}

/// 服务配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// 服务名称
    pub service_name: String,
    /// 服务版本
    pub service_version: String,
    /// 资源配置
    pub resource_config: ResourceConfig,
}

/// 资源配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// CPU配置
    pub cpu_config: CpuConfig,
    /// 内存配置
    pub memory_config: MemoryConfig,
    /// 存储配置
    pub storage_config: StorageConfig,
}

/// CPU配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// CPU核心数
    pub cpu_cores: u32,
    /// CPU频率（GHz）
    pub cpu_frequency_ghz: f64,
}

/// 内存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// 内存大小（GB）
    pub memory_size_gb: f64,
    /// 内存类型
    pub memory_type: String,
}

/// 存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// 存储大小（GB）
    pub storage_size_gb: f64,
    /// 存储类型
    pub storage_type: String,
    /// IOPS
    pub iops: u32,
}

/// AI服务性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIServicePerformanceMetrics {
    /// 模型准确率
    pub model_accuracy: f64,
    /// 推理延迟（毫秒）
    pub inference_latency_ms: f64,
    /// 吞吐量（请求/秒）
    pub throughput_rps: f64,
    /// 资源利用率
    pub resource_utilization: ResourceUtilization,
}

/// 资源利用率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU利用率
    pub cpu_utilization: f64,
    /// 内存利用率
    pub memory_utilization: f64,
    /// GPU利用率
    pub gpu_utilization: Option<f64>,
    /// 存储利用率
    pub storage_utilization: f64,
}

/// 处理时间统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeStats {
    /// 数据加载时间（秒）
    pub data_loading_seconds: f64,
    /// 数据分析时间（秒）
    pub data_analysis_seconds: f64,
    /// 模型生成时间（秒）
    pub model_generation_seconds: f64,
    /// 训练时间（秒）
    pub training_seconds: f64,
    /// 部署时间（秒）
    pub deployment_seconds: f64,
    /// 总处理时间（秒）
    pub total_processing_seconds: f64,
} 
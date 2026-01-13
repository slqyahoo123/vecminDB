/// AI数据库服务配置模块
/// 
/// 包含服务配置、安全配置、性能配置等相关类型定义

use serde::{Serialize, Deserialize};
use crate::compat::data_to_model_engine::ConversionConfig;
use crate::compat::end_to_end_pipeline::PipelineConfig;

/// AI数据库配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIDatabaseConfig {
    /// 服务名称
    pub service_name: String,
    /// 服务版本
    pub service_version: String,
    /// 是否启用自动模式
    pub enable_auto_mode: bool,
    /// 数据转换配置
    pub conversion_config: ConversionConfig,
    /// 流程配置
    pub pipeline_config: PipelineConfig,
    /// 安全配置
    pub security_config: SecurityConfig,
    /// 性能配置
    pub performance_config: PerformanceConfig,
}

/// 安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// 是否启用数据加密
    pub enable_data_encryption: bool,
    /// 是否启用访问控制
    pub enable_access_control: bool,
    /// 是否启用审计日志
    pub enable_audit_logging: bool,
    /// 默认安全级别
    pub default_security_level: String,
}

/// 性能配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// 最大并发请求数
    pub max_concurrent_requests: usize,
    /// 缓存大小（字节）
    pub cache_size_bytes: usize,
    /// 是否启用性能监控
    pub enable_performance_monitoring: bool,
    /// 性能优化级别
    pub optimization_level: OptimizationLevel,
}

/// 优化级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Standard,
    Aggressive,
    Maximum,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_data_encryption: false,
            enable_access_control: false,
            enable_audit_logging: false,
            default_security_level: "basic".to_string(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            cache_size_bytes: 1024 * 1024 * 1024, // 1GB
            enable_performance_monitoring: true,
            optimization_level: OptimizationLevel::Standard,
        }
    }
}

impl Default for AIDatabaseConfig {
    fn default() -> Self {
        Self {
            service_name: "VecMind AI Database".to_string(),
            service_version: "1.0.0".to_string(),
            enable_auto_mode: true,
            conversion_config: ConversionConfig::default(),
            pipeline_config: PipelineConfig::default(),
            security_config: SecurityConfig::default(),
            performance_config: PerformanceConfig::default(),
        }
    }
} 
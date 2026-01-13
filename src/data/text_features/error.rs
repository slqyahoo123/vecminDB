// src/data/text_features/error.rs
//
// Transformer 模型错误类型模块

use std::io;
use thiserror::Error;

/// Transformer模型错误类型
#[derive(Error, Debug)]
pub enum TransformerError {
    /// 模型未初始化
    #[error("模型未初始化")]
    UninitializedModel,
    
    /// IO错误
    #[error("IO错误: {0}")]
    IoError(#[from] io::Error),
    
    /// 解析错误
    #[error("解析错误: {0}")]
    ParseError(String),
    
    /// 输入错误
    #[error("输入错误: {0}")]
    InputError(String),
    
    /// 词汇表错误
    #[error("词汇表错误: {0}")]
    VocabularyError(String),
    
    /// 编码错误
    #[error("编码错误: {0}")]
    EncodingError(String),
    
    /// 特征提取错误
    #[error("特征提取错误: {0}")]
    FeatureExtractionError(String),
    
    /// 模型配置错误
    #[error("模型配置错误: {0}")]
    ConfigError(String),
    
    /// 序列化错误
    #[error("序列化错误: {0}")]
    SerializationError(String),
    
    /// 反序列化错误
    #[error("反序列化错误: {0}")]
    DeserializationError(String),
    
    /// 内存错误
    #[error("内存错误: {0}")]
    MemoryError(String),
    
    /// 计算错误
    #[error("计算错误: {0}")]
    ComputationError(String),
    
    /// 维度不匹配错误
    #[error("维度不匹配: 期望 {expected}, 实际 {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    /// 资源不足错误
    #[error("资源不足: {0}")]
    ResourceError(String),
    
    /// 超时错误
    #[error("操作超时: {0}")]
    TimeoutError(String),
    
    /// 未知错误
    #[error("未知错误: {0}")]
    Unknown(String),
}

impl TransformerError {
    /// 创建词汇表错误
    pub fn vocabulary_error(message: impl Into<String>) -> Self {
        Self::VocabularyError(message.into())
    }
    
    /// 创建编码错误
    pub fn encoding_error(message: impl Into<String>) -> Self {
        Self::EncodingError(message.into())
    }
    
    /// 创建特征提取错误
    pub fn feature_extraction_error(message: impl Into<String>) -> Self {
        Self::FeatureExtractionError(message.into())
    }
    
    /// 创建配置错误
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError(message.into())
    }
    
    /// 创建序列化错误
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError(message.into())
    }
    
    /// 创建反序列化错误
    pub fn deserialization_error(message: impl Into<String>) -> Self {
        Self::DeserializationError(message.into())
    }
    
    /// 创建内存错误
    pub fn memory_error(message: impl Into<String>) -> Self {
        Self::MemoryError(message.into())
    }
    
    /// 创建计算错误
    pub fn computation_error(message: impl Into<String>) -> Self {
        Self::ComputationError(message.into())
    }
    
    /// 创建维度不匹配错误
    pub fn dimension_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
    
    /// 创建资源错误
    pub fn resource_error(message: impl Into<String>) -> Self {
        Self::ResourceError(message.into())
    }
    
    /// 创建超时错误
    pub fn timeout_error(message: impl Into<String>) -> Self {
        Self::TimeoutError(message.into())
    }
    
    /// 创建未知错误
    pub fn unknown(message: impl Into<String>) -> Self {
        Self::Unknown(message.into())
    }
    
    /// 检查是否为致命错误
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::MemoryError(_) | Self::ResourceError(_) | Self::TimeoutError(_)
        )
    }
    
    /// 获取错误代码
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::UninitializedModel => "TRANSFORMER_001",
            Self::IoError(_) => "TRANSFORMER_002",
            Self::ParseError(_) => "TRANSFORMER_003",
            Self::InputError(_) => "TRANSFORMER_004",
            Self::VocabularyError(_) => "TRANSFORMER_005",
            Self::EncodingError(_) => "TRANSFORMER_006",
            Self::FeatureExtractionError(_) => "TRANSFORMER_007",
            Self::ConfigError(_) => "TRANSFORMER_008",
            Self::SerializationError(_) => "TRANSFORMER_009",
            Self::DeserializationError(_) => "TRANSFORMER_010",
            Self::MemoryError(_) => "TRANSFORMER_011",
            Self::ComputationError(_) => "TRANSFORMER_012",
            Self::DimensionMismatch { .. } => "TRANSFORMER_013",
            Self::ResourceError(_) => "TRANSFORMER_014",
            Self::TimeoutError(_) => "TRANSFORMER_015",
            Self::Unknown(_) => "TRANSFORMER_999",
        }
    }
    
    /// 获取错误级别
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::UninitializedModel | Self::InputError(_) => ErrorSeverity::Warning,
            Self::ParseError(_) | Self::ConfigError(_) => ErrorSeverity::Error,
            Self::MemoryError(_) | Self::ResourceError(_) | Self::TimeoutError(_) => ErrorSeverity::Critical,
            _ => ErrorSeverity::Error,
        }
    }
}

/// 错误严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// 警告
    Warning,
    /// 错误
    Error,
    /// 严重错误
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// 错误上下文信息
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// 错误发生的位置
    pub location: String,
    /// 错误发生的时间
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 额外的上下文信息
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// 创建新的错误上下文
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    /// 添加额外信息
    pub fn with_info(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

/// 增强的错误类型，包含上下文信息
#[derive(Debug)]
pub struct TransformerErrorWithContext {
    /// 原始错误
    pub error: TransformerError,
    /// 错误上下文
    pub context: ErrorContext,
}

impl TransformerErrorWithContext {
    /// 创建新的错误
    pub fn new(error: TransformerError, context: ErrorContext) -> Self {
        Self { error, context }
    }
    
    /// 获取错误代码
    pub fn error_code(&self) -> &'static str {
        self.error.error_code()
    }
    
    /// 获取错误严重程度
    pub fn severity(&self) -> ErrorSeverity {
        self.error.severity()
    }
    
    /// 检查是否为致命错误
    pub fn is_fatal(&self) -> bool {
        self.error.is_fatal()
    }
}

impl std::fmt::Display for TransformerErrorWithContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} at {}: {}",
            self.error_code(),
            self.error,
            self.context.location,
            self.context.timestamp
        )
    }
}

impl std::error::Error for TransformerErrorWithContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = TransformerError::vocabulary_error("词汇表加载失败");
        assert_eq!(error.error_code(), "TRANSFORMER_005");
        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(!error.is_fatal());
    }

    #[test]
    fn test_fatal_error() {
        let error = TransformerError::memory_error("内存不足");
        assert_eq!(error.error_code(), "TRANSFORMER_011");
        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert!(error.is_fatal());
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_function")
            .with_info("model_id", "test_model")
            .with_info("operation", "initialization");
        
        assert_eq!(context.location, "test_function");
        assert_eq!(context.additional_info.len(), 2);
    }

    #[test]
    fn test_error_with_context() {
        let error = TransformerError::InputError("无效输入".to_string());
        let context = ErrorContext::new("test_function");
        let error_with_context = TransformerErrorWithContext::new(error, context);
        
        assert_eq!(error_with_context.error_code(), "TRANSFORMER_004");
        assert_eq!(error_with_context.severity(), ErrorSeverity::Warning);
        assert!(!error_with_context.is_fatal());
    }
} 
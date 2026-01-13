/// 统一类型映射
/// 
/// 此文件定义了所有模块间共享的类型别名，
/// 用于解决循环依赖问题，确保类型一致性

// 从算法模块导入执行状态
pub use crate::algorithm::types::ExecutionStatus;

// 重新导出核心类型
pub use crate::core::types::{
    // 张量相关
    CoreTensorData,
    DataType,
    DeviceType,
    
    // 模型相关
    CoreModelParameters,
    ModelStatus,
    ModelInfo,
    
    // 训练相关
    CoreTrainingConfig,
    CoreTrainingResult,
    TrainingStatus,
    OptimizerType,
    LossFunctionType,
    CoreEarlyStoppingConfig,
    
    // 数据相关
    CoreDataBatch,
    CoreDataSchema,
    CoreSchemaField,
    CoreFieldType,
    FieldConstraints,
    
    // 算法相关
    CoreAlgorithmDefinition,
    AlgorithmType,
    CoreAlgorithmParameter,
    ParameterType,
    ResourceRequirements,
    
    // 执行结果相关
    CoreExecutionResult,
    CoreLogEntry,
    LogLevel,
    
    // 系统状态相关
    CoreComponentStatus,
    ComponentType,
    ComponentStatusValue,
    CoreIssue,
    IssueSeverity,
    
    // 数据处理相关
    ProcessedData as CoreProcessedData,
};

// 从接口模块导出验证结果
pub use crate::core::interfaces::ValidationResult;

// 为了向后兼容，创建统一的类型别名
// 这些别名将在所有模块中使用，避免循环依赖

/// 张量数据类型别名
pub type TensorData = CoreTensorData;

/// 模型参数类型别名
pub type ModelParameters = CoreModelParameters;

/// 数据批次类型别名
pub type DataBatch = CoreDataBatch;

/// 数据模式类型别名
pub type DataSchema = CoreDataSchema;

/// 训练配置类型别名
pub type TrainingConfig = CoreTrainingConfig;

/// 训练结果类型别名
pub type TrainingResult = CoreTrainingResult;

// 算法定义类型别名已通过 algorithm/mod.rs 重导出，避免重复定义

/// 算法参数类型别名
pub type AlgorithmParameter = CoreAlgorithmParameter;

/// 执行结果类型别名
pub type ExecutionResult = CoreExecutionResult;

/// 日志条目类型别名
pub type LogEntry = CoreLogEntry;

/// 组件状态类型别名
pub type ComponentStatus = CoreComponentStatus;

/// 系统问题类型别名
pub type Issue = CoreIssue;

/// 模式字段类型别名
pub type SchemaField = CoreSchemaField;

/// 字段类型别名
pub type FieldType = CoreFieldType;

/// 处理数据类型别名
pub type ProcessedData = CoreProcessedData;

// ValidationResult 已在 core::interfaces 中导入，无需重复定义

// 导出常用的构造函数和工具函数
impl TensorData {
    /// 创建新的张量数据
    pub fn new_unified(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self::new(shape, data)
    }
    
    /// 创建零张量
    pub fn zeros_unified(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(shape, vec![0.0; size])
    }
    
    /// 创建一张量
    pub fn ones_unified(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(shape, vec![1.0; size])
    }
}

impl ModelParameters {
    /// 创建新的模型参数
    pub fn new_unified(parameters: std::collections::HashMap<String, TensorData>) -> Self {
        Self::new(parameters)
    }
    
    /// 从张量映射创建（便捷方法）
    pub fn from_tensors(tensors: std::collections::HashMap<String, TensorData>) -> Self {
        Self::new(tensors)
    }
    
    /// 创建空的模型参数
    pub fn empty() -> Self {
        Self::new(std::collections::HashMap::new())
    }
}

impl DataBatch {
    /// 创建新的数据批次
    pub fn new_unified(data: Vec<TensorData>, labels: Option<Vec<TensorData>>) -> Self {
        Self::new(data, labels)
    }
    
    /// 从单个张量创建
    pub fn from_tensor(tensor: TensorData) -> Self {
        Self::new(vec![tensor], None)
    }
    
    /// 创建空批次
    pub fn empty() -> Self {
        Self::new(Vec::new(), None)
    }
}

impl TrainingConfig {
    /// 创建新的训练配置（便捷方法，调用 CoreTrainingConfig::new）
    pub fn new_unified(model_id: String, dataset_id: String) -> Self {
        Self::new(model_id, dataset_id)
    }
    
    // 注意：with_learning_rate, with_batch_size, with_epochs 已在 core/types.rs 的 CoreTrainingConfig impl 中定义，这里删除重复定义
}

impl CoreAlgorithmDefinition {
    /// 创建新的算法定义
    pub fn new_unified(id: String, name: String, algorithm_type: AlgorithmType) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            name,
            algorithm_type,
            parameters: Vec::new(),
            description: String::new(),
            version: "1.0.0".to_string(),
            source_code: String::new(),
            language: "rust".to_string(),
            input_schema: Vec::new(),
            output_schema: Vec::new(),
            resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
                max_memory_mb: 512,
                max_cpu_percent: 100.0,
                max_execution_time_seconds: 3600,
                requires_gpu: false,
                max_gpu_memory_mb: None,
                network_access: false,
                file_system_access: Vec::new(),
            },
            metadata: std::collections::HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
    
    /// 添加整数参数
    pub fn add_int_parameter(mut self, name: String, default: Option<i64>) -> Self {
        let param = AlgorithmParameter {
            name,
            parameter_type: ParameterType::Integer,
            required: false,
            default_value: default.map(|v| v.to_string()),
            description: String::new(),
        };
        self.parameters.push(param);
        self
    }
    
    /// 添加浮点参数
    pub fn add_float_parameter(mut self, name: String, default: Option<f64>) -> Self {
        let param = AlgorithmParameter {
            name,
            parameter_type: ParameterType::Float,
            required: false,
            default_value: default.map(|v| v.to_string()),
            description: String::new(),
        };
        self.parameters.push(param);
        self
    }
}

// 提供模块间通信的统一接口
pub mod communication {
    use super::*;
    use async_trait::async_trait;
    
    /// 模块间数据传输接口
    #[async_trait]
    pub trait DataTransfer: Send + Sync {
        /// 发送数据批次
        async fn send_batch(&self, batch: DataBatch) -> crate::Result<()>;
        
        /// 接收数据批次
        async fn receive_batch(&self) -> crate::Result<Option<DataBatch>>;
        
        /// 发送模型参数
        async fn send_parameters(&self, params: ModelParameters) -> crate::Result<()>;
        
        /// 接收模型参数
        async fn receive_parameters(&self) -> crate::Result<Option<ModelParameters>>;
    }
    
    /// 模块间事件通信接口
    #[async_trait]
    pub trait EventCommunication: Send + Sync {
        /// 发布事件
        async fn publish_event(&self, event_type: String, data: serde_json::Value) -> crate::Result<()>;
        
        /// 订阅事件
        async fn subscribe_event(&self, event_type: String) -> crate::Result<()>;
        
        /// 取消订阅
        async fn unsubscribe_event(&self, event_type: String) -> crate::Result<()>;
    }
    
    /// 模块状态同步接口
    #[async_trait]
    pub trait StatusSync: Send + Sync {
        /// 更新组件状态
        async fn update_status(&self, status: ComponentStatus) -> crate::Result<()>;
        
        /// 获取组件状态
        async fn get_status(&self, component: ComponentType) -> crate::Result<Option<ComponentStatus>>;
        
        /// 报告问题
        async fn report_issue(&self, issue: Issue) -> crate::Result<()>;
        
        /// 解决问题
        async fn resolve_issue(&self, issue_id: String) -> crate::Result<()>;
    }
}

// 提供类型转换工具
pub mod conversion {
    use super::*;
    
    /// 类型转换错误
    #[derive(Debug, thiserror::Error)]
    pub enum ConversionError {
        #[error("形状不匹配: 期望 {expected:?}, 实际 {actual:?}")]
        ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
        
        #[error("数据类型不匹配: 期望 {expected}, 实际 {actual}")]
        TypeMismatch { expected: String, actual: String },
        
        #[error("数据大小不匹配: 期望 {expected}, 实际 {actual}")]
        SizeMismatch { expected: usize, actual: usize },
        
        #[error("转换失败: {message}")]
        ConversionFailed { message: String },
    }
    
    /// 张量转换工具
    pub struct TensorConverter;
    
    impl TensorConverter {
        /// 转换为指定形状
        pub fn reshape(tensor: &mut TensorData, new_shape: Vec<usize>) -> Result<(), ConversionError> {
            let expected_size: usize = new_shape.iter().product();
            let actual_size = tensor.size();
            
            if expected_size != actual_size {
                return Err(ConversionError::SizeMismatch {
                    expected: expected_size,
                    actual: actual_size,
                });
            }
            
            tensor.reshape(new_shape).map_err(|e| ConversionError::ConversionFailed {
                message: e,
            })
        }
        
        /// 转换数据类型
        pub fn convert_dtype(tensor: &mut TensorData, target_type: String) -> Result<(), ConversionError> {
            if tensor.dtype == target_type {
                return Ok(());
            }
            
            // 这里可以实现实际的数据类型转换逻辑
            tensor.dtype = target_type;
            Ok(())
        }
        
        /// 批量转换张量
        pub fn batch_convert(
            tensors: &mut [TensorData],
            target_shape: Option<Vec<usize>>,
            target_type: Option<String>,
        ) -> Result<(), ConversionError> {
            for tensor in tensors {
                if let Some(ref shape) = target_shape {
                    Self::reshape(tensor, shape.clone())?;
                }
                if let Some(ref dtype) = target_type {
                    Self::convert_dtype(tensor, dtype.clone())?;
                }
            }
            Ok(())
        }
    }
    
    /// 数据批次转换工具
    pub struct BatchConverter;
    
    impl BatchConverter {
        /// 合并多个数据批次
        pub fn merge_batches(batches: Vec<DataBatch>) -> Result<DataBatch, ConversionError> {
            if batches.is_empty() {
                return Err(ConversionError::ConversionFailed {
                    message: "无法合并空的批次列表".to_string(),
                });
            }
            
            let mut merged = batches.into_iter().next().unwrap();
            // 这里可以实现实际的批次合并逻辑
            
            Ok(merged)
        }
        
        /// 分割数据批次
        pub fn split_batch(batch: DataBatch, split_size: usize) -> Result<Vec<DataBatch>, ConversionError> {
            if split_size == 0 {
                return Err(ConversionError::ConversionFailed {
                    message: "分割大小不能为0".to_string(),
                });
            }
            
            // 这里可以实现实际的批次分割逻辑
            Ok(vec![batch])
        }
    }
}

// 提供验证工具
pub mod validation {
    use super::*;
    
    /// 验证错误
    #[derive(Debug, thiserror::Error)]
    pub enum ValidationError {
        #[error("张量验证失败: {message}")]
        TensorValidation { message: String },
        
        #[error("批次验证失败: {message}")]
        BatchValidation { message: String },
        
        #[error("配置验证失败: {message}")]
        ConfigValidation { message: String },
    }
    
    /// 张量验证器
    pub struct TensorValidator;
    
    impl TensorValidator {
        /// 验证张量有效性
        pub fn validate(tensor: &TensorData) -> Result<(), ValidationError> {
            // 检查形状
            if tensor.shape.is_empty() {
                return Err(ValidationError::TensorValidation {
                    message: "张量形状不能为空".to_string(),
                });
            }
            
            // 检查数据大小
            let expected_size: usize = tensor.shape.iter().product();
            if tensor.data.len() != expected_size {
                return Err(ValidationError::TensorValidation {
                    message: format!(
                        "数据大小不匹配: 期望 {}, 实际 {}",
                        expected_size,
                        tensor.data.len()
                    ),
                });
            }
            
            // 检查数据有效性
            for (i, &value) in tensor.data.iter().enumerate() {
                if value.is_nan() || value.is_infinite() {
                    return Err(ValidationError::TensorValidation {
                        message: format!("数据包含无效值 (NaN/Inf) 在位置 {}", i),
                    });
                }
            }
            
            Ok(())
        }
        
        /// 验证张量兼容性
        pub fn validate_compatibility(tensor1: &TensorData, tensor2: &TensorData) -> Result<(), ValidationError> {
            if tensor1.dtype != tensor2.dtype {
                return Err(ValidationError::TensorValidation {
                    message: format!(
                        "数据类型不兼容: {:?} vs {:?}",
                        tensor1.dtype, tensor2.dtype
                    ),
                });
            }
            
            if tensor1.device != tensor2.device {
                return Err(ValidationError::TensorValidation {
                    message: format!(
                        "设备类型不兼容: {:?} vs {:?}",
                        tensor1.device, tensor2.device
                    ),
                });
            }
            
            Ok(())
        }
    }
    
    /// 批次验证器
    pub struct BatchValidator;
    
    impl BatchValidator {
        /// 验证数据批次
        pub fn validate(batch: &DataBatch) -> Result<(), ValidationError> {
            // 检查批次大小
            if batch.batch_size == 0 {
                return Err(ValidationError::BatchValidation {
                    message: "批次大小不能为0".to_string(),
                });
            }
            
            // 验证数据张量
            for tensor in &batch.data {
                TensorValidator::validate(tensor).map_err(|e| ValidationError::BatchValidation {
                    message: format!("数据张量验证失败: {}", e),
                })?;
                
                // 检查批次维度
                if tensor.shape.is_empty() || tensor.shape[0] != batch.batch_size {
                    return Err(ValidationError::BatchValidation {
                        message: format!(
                            "数据张量的批次维度不匹配: 期望 {}, 实际 {}",
                            batch.batch_size,
                            tensor.shape.get(0).unwrap_or(&0)
                        ),
                    });
                }
            }
            
            // 验证标签张量（如果存在）
            if let Some(ref labels) = batch.labels {
                for tensor in labels {
                    TensorValidator::validate(tensor).map_err(|e| ValidationError::BatchValidation {
                        message: format!("标签张量验证失败: {}", e),
                    })?;
                }
            }
            
            Ok(())
        }
    }
} 
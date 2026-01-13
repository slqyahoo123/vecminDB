// 模型相关类型定义

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// 可序列化的模型状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableModelStatus {
    Created,
    Training,
    Trained,
    Deployed,
    Error,
}

/// 监控状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringStatus {
    Running,
    Stopped,
    Error,
}

/// A/B测试状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ABTestStatus {
    Running,
    Completed,
    Stopped,
}

/// 部署状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Deploying,
    Deployed,
    Failed,
    Undeployed,
}

/// 可序列化的简化Model结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableModel {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub version: String,
    pub model_type: String,
    pub status: SerializableModelStatus,
    pub metadata: HashMap<String, String>,
    pub created_at: i64, // 使用timestamp代替DateTime
    pub updated_at: i64,
    pub parent_id: Option<String>,
}

impl From<&crate::model::Model> for SerializableModel {
    fn from(model: &crate::model::Model) -> Self {
        Self {
            id: model.id.clone(),
            name: model.name.clone(),
            description: model.description.clone(),
            version: model.version.clone(),
            model_type: model.model_type.clone(),
            status: SerializableModelStatus::Created, // 简化状态映射
            metadata: model.metadata.clone(),
            created_at: model.created_at.timestamp(),
            updated_at: model.updated_at.timestamp(),
            parent_id: model.parent_id.clone(),
        }
    }
}

impl SerializableModel {
    pub fn to_full_model(&self) -> crate::model::Model {
        use std::sync::{Arc, Mutex};
        
        crate::model::Model {
            id: self.id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            version: self.version.clone(),
            model_type: self.model_type.clone(),
            smart_parameters: crate::model::SmartModelParameters::default(),
            architecture: crate::model::ModelArchitecture::default(),
            status: crate::model::ModelStatus::Created,
            metrics: None,
            created_at: chrono::DateTime::from_timestamp(self.created_at, 0).unwrap_or_else(chrono::Utc::now),
            updated_at: chrono::DateTime::from_timestamp(self.updated_at, 0).unwrap_or_else(chrono::Utc::now),
            parent_id: self.parent_id.clone(),
            metadata: self.metadata.clone(),
            memory_monitor: Arc::new(Mutex::new(crate::model::ModelMemoryMonitor::new())),
        }
    }
}

/// 训练过程中使用的模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 模型ID
    pub id: String,
    /// 模型名称
    pub name: String,
    /// 模型类型
    pub model_type: String,
    /// 算法名称
    pub algorithm_name: String,
    /// 参数数量
    pub parameter_count: Option<usize>,
    /// 输入维度
    pub input_dimensions: Option<Vec<usize>>,
    /// 输出维度
    pub output_dimensions: Option<Vec<usize>>,
    /// 模型架构信息
    pub architecture_info: Option<ModelArchitectureInfo>,
    /// 模型元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间
    pub created_at: i64,
    /// 更新时间
    pub updated_at: i64,
}

/// 模型架构信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitectureInfo {
    /// 层数
    pub num_layers: usize,
    /// 隐藏层大小
    pub hidden_sizes: Vec<usize>,
    /// 激活函数类型
    pub activation_functions: Vec<String>,
    /// 是否使用dropout
    pub uses_dropout: bool,
    /// dropout率
    pub dropout_rate: Option<f32>,
    /// 是否使用批归一化
    pub uses_batch_norm: bool,
    /// 其他架构参数
    pub additional_params: HashMap<String, serde_json::Value>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            name: "默认模型".to_string(),
            model_type: "neural_network".to_string(),
            algorithm_name: "default".to_string(),
            parameter_count: None,
            input_dimensions: None,
            output_dimensions: None,
            architecture_info: None,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
        }
    }
}

impl ModelInfo {
    /// 创建新的模型信息
    pub fn new(id: String, name: String, model_type: String, algorithm_name: String) -> Self {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
            
        Self {
            id,
            name,
            model_type,
            algorithm_name,
            parameter_count: None,
            input_dimensions: None,
            output_dimensions: None,
            architecture_info: None,
            metadata: HashMap::new(),
            created_at: current_time,
            updated_at: current_time,
        }
    }
    
    /// 获取参数数量的估算值
    pub fn estimate_parameter_count(&self) -> usize {
        if let Some(count) = self.parameter_count {
            return count;
        }
        
        // 基于架构信息估算参数数量
        if let Some(arch_info) = &self.architecture_info {
            if !arch_info.hidden_sizes.is_empty() {
                // 简单的参数数量估算：层之间的连接权重
                let mut total_params = 0;
                for i in 0..arch_info.hidden_sizes.len() {
                    if i == 0 {
                        // 输入层到第一个隐藏层
                        if let Some(input_dims) = &self.input_dimensions {
                            total_params += input_dims.iter().product::<usize>() * arch_info.hidden_sizes[i];
                        }
                    } else {
                        // 隐藏层之间
                        total_params += arch_info.hidden_sizes[i - 1] * arch_info.hidden_sizes[i];
                    }
                    
                    // 偏置项
                    total_params += arch_info.hidden_sizes[i];
                }
                
                // 最后一层到输出层
                if let Some(output_dims) = &self.output_dimensions {
                    if let Some(last_hidden) = arch_info.hidden_sizes.last() {
                        total_params += last_hidden * output_dims.iter().product::<usize>();
                        total_params += output_dims.iter().product::<usize>(); // 输出层偏置
                    }
                }
                
                return total_params;
            }
        }
        
        // 默认估算值
        match self.model_type.as_str() {
            "linear" => 1000,
            "mlp" => 10000,
            "cnn" => 100000,
            "transformer" => 1000000,
            _ => 10000,
        }
    }
}

// 类型别名
pub type ModelHealthStatus = crate::model::manager::metrics::ModelHealthStatus;
pub type ModelPerformanceMetrics = crate::model::manager::metrics::ModelPerformanceMetrics;
pub type MonitoringConfig = crate::model::manager::config::MonitoringConfig;
pub type ModelMonitoringConfig = crate::model::manager::metrics::ModelMonitoringConfig;
pub type CompressionConfig = crate::model::manager::config::CompressionConfig;
pub type QuantizationConfig = crate::model::manager::config::QuantizationConfig;
pub type OptimizationConfig = crate::model::manager::config::OptimizationConfig;
pub type IntegrityCheckResult = crate::model::manager::metrics::IntegrityCheckResult;
pub type ABTestConfig = crate::model::manager::config::ABTestConfig;
pub type ABTestResults = crate::model::manager::metrics::ABTestResults;
pub type DeploymentConfig = crate::model::manager::config::DeploymentConfig; 
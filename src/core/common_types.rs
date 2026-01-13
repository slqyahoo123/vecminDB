/// 核心公共类型定义
/// 所有模块共享的基础数据类型，避免循环依赖

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::ops::{Add, Sub, Mul, Div, AddAssign};
use crate::core::types::{DataType, DeviceType};

// 直接定义UnifiedTensorData，避免循环导入
/// 统一的张量数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub dtype: UnifiedDataType,
    pub device: String,
}

impl UnifiedTensorData {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            shape,
            data,
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_size = shape.iter().product::<usize>();
        Self {
            shape,
            data: vec![0.0; total_size],
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let total_size = shape.iter().product::<usize>();
        Self {
            shape,
            data: vec![1.0; total_size],
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }
    }

    pub fn random(shape: Vec<usize>) -> Self {
        let total_size = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(total_size);
        // 使用简单的伪随机数生成，避免外部依赖
        let mut seed = 12345u64;
        for _ in 0..total_size {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((seed % 1000) as f32 / 1000.0);
        }
        Self {
            shape,
            data,
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }
    }

    pub fn validate(&self) -> bool {
        let expected_size = self.shape.iter().product::<usize>();
        self.data.len() == expected_size
    }
}

impl Default for UnifiedTensorData {
    fn default() -> Self {
        Self {
            shape: vec![1],
            data: vec![0.0],
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }
    }
}

impl Add for UnifiedTensorData {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "张量形状必须相同才能相加");
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Add<f32> for UnifiedTensorData {
    type Output = Self;

    fn add(self, scalar: f32) -> Self::Output {
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x + scalar)
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl AddAssign for UnifiedTensorData {
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.shape, other.shape, "张量形状必须相同才能相加");
        
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }
}

impl AddAssign<f32> for UnifiedTensorData {
    fn add_assign(&mut self, scalar: f32) {
        for x in self.data.iter_mut() {
            *x += scalar;
        }
    }
}

impl Sub for UnifiedTensorData {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "张量形状必须相同才能相减");
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Mul for UnifiedTensorData {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "张量形状必须相同才能相乘");
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Mul<f32> for UnifiedTensorData {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x * scalar)
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Div for UnifiedTensorData {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "张量形状必须相同才能相除");
        
        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                assert!(*b != 0.0, "除数不能为零");
                a / b
            })
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Div<f32> for UnifiedTensorData {
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output {
        assert!(scalar != 0.0, "除数不能为零");
        
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x / scalar)
            .collect();
        
        UnifiedTensorData {
            shape: self.shape,
            data: result_data,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

/// 统一的数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedDataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    String,
}

impl UnifiedDataType {
    /// 转换为字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            UnifiedDataType::Float32 => "float32",
            UnifiedDataType::Float64 => "float64",
            UnifiedDataType::Int32 => "int32",
            UnifiedDataType::Int64 => "int64",
            UnifiedDataType::Bool => "bool",
            UnifiedDataType::String => "string",
        }
    }
}

/// 统一的设备类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedDeviceType {
    CPU,
    GPU(u32),
    TPU(u32),
    Custom(String),
}

impl UnifiedDeviceType {
    /// 转换为字符串表示
    pub fn to_string(&self) -> String {
        match self {
            UnifiedDeviceType::CPU => "cpu".to_string(),
            UnifiedDeviceType::GPU(id) => format!("gpu:{}", id),
            UnifiedDeviceType::TPU(id) => format!("tpu:{}", id),
            UnifiedDeviceType::Custom(name) => name.clone(),
        }
    }
    
    /// 从字符串解析设备类型
    pub fn from_string(s: &str) -> Self {
        if s == "cpu" {
            UnifiedDeviceType::CPU
        } else if s.starts_with("gpu:") {
            let id = s.strip_prefix("gpu:").unwrap_or("0").parse().unwrap_or(0);
            UnifiedDeviceType::GPU(id)
        } else if s.starts_with("tpu:") {
            let id = s.strip_prefix("tpu:").unwrap_or("0").parse().unwrap_or(0);
            UnifiedDeviceType::TPU(id)
        } else {
            UnifiedDeviceType::Custom(s.to_string())
        }
    }
}

impl std::fmt::Display for UnifiedDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl From<UnifiedDeviceType> for String {
    fn from(device: UnifiedDeviceType) -> Self {
        device.to_string()
    }
}

impl From<&UnifiedDeviceType> for String {
    fn from(device: &UnifiedDeviceType) -> Self {
        device.to_string()
    }
}

impl From<String> for UnifiedDeviceType {
    fn from(s: String) -> Self {
        UnifiedDeviceType::from_string(&s)
    }
}

impl From<&str> for UnifiedDeviceType {
    fn from(s: &str) -> Self {
        UnifiedDeviceType::from_string(s)
    }
}

/// 统一的模型参数结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedModelParameters {
    pub version: String,
    pub parameters: HashMap<String, UnifiedTensorData>,
    pub metadata: HashMap<String, String>,
    pub checksum: Option<String>,
}

/// 统一的数据批次结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDataBatch {
    pub inputs: Vec<UnifiedTensorData>,
    pub targets: Option<Vec<UnifiedTensorData>>,
    pub labels: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub batch_size: usize,
}

/// 统一的执行结果结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExecutionResult {
    pub success: bool,
    pub outputs: Vec<UnifiedTensorData>,
    pub metrics: HashMap<String, f32>,
    pub execution_time_ms: u64,
    pub resource_usage: UnifiedResourceUsage,
    pub error_message: Option<String>,
}

/// 统一的资源使用情况
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedResourceUsage {
    pub memory_bytes: usize,
    pub cpu_time_ms: u64,
    pub gpu_time_ms: u64,
    pub disk_io_bytes: usize,
    pub network_io_bytes: usize,
}

/// 统一的训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTrainingConfig {
    pub model_id: String,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub device: UnifiedDeviceType,
    pub metadata: HashMap<String, String>,
}

/// 统一的算法定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAlgorithmDefinition {
    pub id: String,
    pub name: String,
    pub algorithm_type: String,
    pub parameters: HashMap<String, String>,
    pub resource_requirements: UnifiedResourceUsage,
    pub metadata: HashMap<String, String>,
}

/// 统一的早停配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedEarlyStoppingConfig {
    pub metric: String,
    pub patience: usize,
    pub min_delta: f32,
    pub mode: String, // "min" or "max"
}

/// 统一的分布式配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDistributedConfig {
    pub world_size: usize,
    pub rank: usize,
    pub backend: String,
    pub master_addr: String,
    pub master_port: u16,
}

/// 统一的算法参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAlgorithmParameter {
    pub name: String,
    pub param_type: String,
    pub default_value: Option<String>,
    pub constraints: Option<UnifiedParameterConstraints>,
}

/// 统一的参数约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedParameterConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub allowed_values: Option<Vec<String>>,
    pub pattern: Option<String>,
}

/// 统一的资源需求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResourceRequirements {
    pub max_memory_mb: usize,
    pub max_cpu_time_seconds: u64,
    pub requires_gpu: bool,
    pub max_disk_space_mb: usize,
    pub max_network_bandwidth_mbps: Option<u32>,
}

/// 类型转换trait
pub trait ToUnified<T> {
    fn to_unified(&self) -> T;
}

pub trait FromUnified<T> {
    fn from_unified(unified: &T) -> Self;
}

impl UnifiedDataBatch {
    pub fn new(inputs: Vec<UnifiedTensorData>) -> Self {
        Self {
            batch_size: inputs.get(0).map_or(0, |t| t.shape[0]),
            inputs,
            targets: None,
            labels: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_targets(mut self, targets: Vec<UnifiedTensorData>) -> Self {
        self.targets = Some(targets);
        self
    }

    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// 获取批次中的样本数量
    pub fn len(&self) -> usize {
        if self.batch_size > 0 {
            self.batch_size
        } else {
            self.inputs.len()
        }
    }
    
    /// 检查批次是否为空
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
    
    /// 获取指定索引的样本（简化实现）
    pub fn get(&self, index: usize) -> Option<HashMap<String, serde_json::Value>> {
        if index < self.inputs.len() {
            let mut sample = HashMap::new();
            
            // 添加输入数据
            if let Some(input) = self.inputs.get(index) {
                sample.insert("input".to_string(), serde_json::json!(input.data));
            }
            
            // 添加目标数据（如果存在）
            if let Some(targets) = &self.targets {
                if let Some(target) = targets.get(index) {
                    sample.insert("label".to_string(), serde_json::json!(target.data));
                }
            }
            
            Some(sample)
        } else {
            None
        }
    }

    /// 从单个张量创建批次
    pub fn from_tensor(tensor: UnifiedTensorData) -> Self {
        Self {
            batch_size: tensor.shape[0],
            inputs: vec![tensor],
            targets: None,
            labels: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// 从DataBatch转换为UnifiedDataBatch
    pub fn from_data_batch(data_batch: crate::data::exports::DataBatch) -> Result<Self, crate::error::Error> {
        let mut inputs = Vec::new();
        let mut targets = None;
        let mut labels = Vec::new();
        let mut metadata = HashMap::new();

        // 转换输入数据
        for (i, row) in data_batch.data.iter().enumerate() {
            let input_tensor = UnifiedTensorData {
                shape: vec![row.len()], // 特征维度
                data: row.clone(), // 使用实际特征数据
                dtype: UnifiedDataType::Float32,
                device: "CPU".to_string(),
            };
            inputs.push(input_tensor);
        }

        // 获取批次大小
        let batch_size = data_batch.data.len();

        // 添加元数据
        metadata.insert("source".to_string(), "DataBatch".to_string());
        metadata.insert("batch_id".to_string(), data_batch.id.clone());

        Ok(Self {
            inputs,
            targets,
            labels,
            metadata,
            batch_size,
        })
    }

    /// 从数据模块的批次类型转换（crate::data::batch::DataBatch -> UnifiedDataBatch）
    /// 生产级实现：尽可能保留元数据与批次大小
    pub fn from_manager_batch(batch: &crate::data::batch::DataBatch) -> Result<Self, crate::error::Error> {
        // 提取特征与标签
        let features = batch.get_features()?; // Vec<Vec<f32>>
        let labels_vecs = batch.get_labels()?; // Vec<Vec<f32>>

        // 构造输入张量：每条样本一条一维张量
        let mut inputs: Vec<UnifiedTensorData> = Vec::with_capacity(features.len());
        for row in &features {
            inputs.push(UnifiedTensorData {
                shape: vec![row.len()],
                data: row.clone(),
                dtype: UnifiedDataType::Float32,
                device: "CPU".to_string(),
            });
        }

        // 目标张量（若存在则与inputs对齐），简单展平为一维向量目标
        let targets = if !labels_vecs.is_empty() {
            let mut t: Vec<UnifiedTensorData> = Vec::with_capacity(labels_vecs.len());
            for row in &labels_vecs {
                t.push(UnifiedTensorData {
                    shape: vec![row.len()],
                    data: row.clone(),
                    dtype: UnifiedDataType::Float32,
                    device: "CPU".to_string(),
                });
            }
            Some(t)
        } else {
            None
        };

        // 复制元数据
        let metadata = batch.metadata.clone();
        let batch_size = features.len();

        Ok(UnifiedDataBatch { inputs, targets, labels: Vec::new(), metadata, batch_size })
    }
}

impl UnifiedExecutionResult {
    pub fn success(outputs: Vec<UnifiedTensorData>) -> Self {
        Self {
            success: true,
            outputs,
            metrics: HashMap::new(),
            execution_time_ms: 0,
            resource_usage: UnifiedResourceUsage::default(),
            error_message: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            outputs: vec![],
            metrics: HashMap::new(),
            execution_time_ms: 0,
            resource_usage: UnifiedResourceUsage::default(),
            error_message: Some(message),
        }
    }

    pub fn with_metrics(mut self, metrics: HashMap<String, f32>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_resource_usage(mut self, usage: UnifiedResourceUsage) -> Self {
        self.resource_usage = usage;
        self
    }
}

/// 统一的性能指标结构
/// 避免在不同模块中重复定义PerformanceMetrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedPerformanceMetrics {
    /// 组件或实体ID
    pub component_id: String,
    /// 准确率
    pub accuracy: Option<f32>,
    /// 精确率
    pub precision: Option<f32>,
    /// 召回率
    pub recall: Option<f32>,
    /// F1分数
    pub f1_score: Option<f32>,
    /// 均方误差
    pub mse: Option<f32>,
    /// 均方根误差
    pub rmse: Option<f32>,
    /// 平均绝对误差
    pub mae: Option<f32>,
    /// R²决定系数
    pub r_squared: Option<f32>,
    /// 推理延迟（毫秒）
    pub inference_latency_ms: Option<f64>,
    /// 吞吐量（请求/秒）
    pub throughput_rps: Option<f64>,
    /// 内存使用量（MB）
    pub memory_usage_mb: Option<f64>,
    /// CPU使用率（0-1）
    pub cpu_utilization: Option<f64>,
    /// GPU使用率（0-1）
    pub gpu_utilization: Option<f64>,
    /// 处理时间（毫秒）
    pub processing_time_ms: Option<u64>,
    /// 自定义指标
    pub custom_metrics: HashMap<String, f32>,
    /// 时间戳
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl UnifiedPerformanceMetrics {
    pub fn new(component_id: String) -> Self {
        Self {
            component_id,
            accuracy: None,
            precision: None,
            recall: None,
            f1_score: None,
            mse: None,
            rmse: None,
            mae: None,
            r_squared: None,
            inference_latency_ms: None,
            throughput_rps: None,
            memory_usage_mb: None,
            cpu_utilization: None,
            gpu_utilization: None,
            processing_time_ms: None,
            custom_metrics: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    pub fn with_precision(mut self, precision: f32) -> Self {
        self.precision = Some(precision);
        self
    }

    pub fn with_recall(mut self, recall: f32) -> Self {
        self.recall = Some(recall);
        self
    }

    pub fn with_f1_score(mut self, f1_score: f32) -> Self {
        self.f1_score = Some(f1_score);
        self
    }

    pub fn with_mse(mut self, mse: f32) -> Self {
        self.mse = Some(mse);
        self
    }

    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.inference_latency_ms = Some(latency_ms);
        self
    }

    pub fn with_throughput(mut self, throughput_rps: f64) -> Self {
        self.throughput_rps = Some(throughput_rps);
        self
    }

    pub fn with_custom_metric(mut self, name: String, value: f32) -> Self {
        self.custom_metrics.insert(name, value);
        self
    }
}

/// 性能指标类型别名，用于向后兼容
pub type PerformanceMetrics = UnifiedPerformanceMetrics;

/// 模型架构定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// 架构ID
    pub id: String,
    /// 架构名称
    pub name: String,
    /// 网络层定义
    pub layers: Vec<LayerDefinition>,
    /// 激活函数
    pub activation: Option<String>,
    /// 优化器类型
    pub optimizer: Option<String>,
    /// 损失函数
    pub loss: Option<String>,
    /// 输入形状
    pub input_shape: Vec<usize>,
    /// 输出形状
    pub output_shape: Vec<usize>,
    /// 参数数量
    pub num_parameters: Option<usize>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
}

/// 网络层定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDefinition {
    /// 层ID
    pub id: String,
    /// 层名称
    pub name: String,
    /// 层类型
    pub layer_type: String,
    /// 层参数
    pub parameters: HashMap<String, serde_json::Value>,
    /// 输入形状
    pub input_shape: Option<Vec<usize>>,
    /// 输出形状
    pub output_shape: Option<Vec<usize>>,
    /// 是否可训练
    pub trainable: bool,
}

impl ModelArchitecture {
    /// 创建新的模型架构
    pub fn new(id: String, name: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            layers: Vec::new(),
            activation: None,
            optimizer: None,
            loss: None,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            num_parameters: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 添加层
    pub fn add_layer(mut self, layer: LayerDefinition) -> Self {
        self.layers.push(layer);
        self.updated_at = Utc::now();
        self
    }

    /// 设置激活函数
    pub fn with_activation(mut self, activation: String) -> Self {
        self.activation = Some(activation);
        self.updated_at = Utc::now();
        self
    }

    /// 设置优化器
    pub fn with_optimizer(mut self, optimizer: String) -> Self {
        self.optimizer = Some(optimizer);
        self.updated_at = Utc::now();
        self
    }

    /// 设置损失函数
    pub fn with_loss(mut self, loss: String) -> Self {
        self.loss = Some(loss);
        self.updated_at = Utc::now();
        self
    }

    /// 设置输入形状
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self.updated_at = Utc::now();
        self
    }

    /// 设置输出形状
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self.updated_at = Utc::now();
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self.updated_at = Utc::now();
        self
    }

    /// 计算参数数量
    pub fn calculate_parameters(&mut self) {
        let mut total_params = 0;
        
        for layer in &self.layers {
            if let Some(params) = layer.parameters.get("num_parameters") {
                if let Some(num) = params.as_u64() {
                    total_params += num as usize;
                }
            }
        }
        
        self.num_parameters = Some(total_params);
        self.updated_at = Utc::now();
    }

    /// 验证架构完整性
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("架构名称不能为空".to_string());
        }

        if self.layers.is_empty() {
            return Err("至少需要一个网络层".to_string());
        }

        if self.input_shape.is_empty() {
            return Err("必须指定输入形状".to_string());
        }

        if self.output_shape.is_empty() {
            return Err("必须指定输出形状".to_string());
        }

        Ok(())
    }
}

impl LayerDefinition {
    /// 创建新的层定义
    pub fn new(id: String, name: String, layer_type: String) -> Self {
        Self {
            id,
            name,
            layer_type,
            parameters: HashMap::new(),
            input_shape: None,
            output_shape: None,
            trainable: true,
        }
    }

    /// 添加参数
    pub fn with_parameter(mut self, key: String, value: serde_json::Value) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// 设置输入形状
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = Some(shape);
        self
    }

    /// 设置输出形状
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = Some(shape);
        self
    }

    /// 设置是否可训练
    pub fn with_trainable(mut self, trainable: bool) -> Self {
        self.trainable = trainable;
        self
    }
} 

impl From<&crate::data::exports::DataBatch> for crate::core::types::CoreDataBatch {
    fn from(batch: &crate::data::exports::DataBatch) -> Self {
        // 构造特征张量：按行拼接为 [batch_size, feature_dim]
        let batch_size = batch.data.len();
        let feature_dim = if batch_size > 0 { batch.data[0].len() } else { 0 };
        let mut features_flat: Vec<f32> = Vec::with_capacity(batch_size.saturating_mul(feature_dim));
        for row in &batch.data {
            features_flat.extend_from_slice(row);
        }

        let features_tensor = crate::core::types::CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: vec![batch_size, feature_dim],
            data: features_flat,
            dtype: format!("{:?}", DataType::Float32),
            device: format!("{:?}", DeviceType::CPU),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let labels_tensor = if !batch.labels.is_empty() {
            Some(crate::core::types::CoreTensorData {
                id: uuid::Uuid::new_v4().to_string(),
                shape: vec![batch.labels.len()],
                data: batch.labels.clone(),
                dtype: format!("{:?}", crate::core::types::DataType::Float32),
                device: format!("{:?}", crate::core::types::DeviceType::CPU),
                requires_grad: false,
                metadata: HashMap::new(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
        } else {
            None
        };

        let mut data_tensors = vec![features_tensor];
        if let Some(ref lbl) = labels_tensor {
            data_tensors.push(lbl.clone());
        }

        crate::core::types::CoreDataBatch {
            id: batch.id.clone(),
            data: data_tensors,
            labels: labels_tensor.map(|t| vec![t]), // 将单个 tensor 包装成 Vec
            batch_size: batch.batch_size,
            metadata: Some(batch.metadata.clone()),
            created_at: batch.created_at,
            updated_at: Utc::now(),
        }
    }
}

impl From<crate::data::processor::types::ProcessorBatch> for crate::core::interfaces::ProcessedData {
    fn from(batch: crate::data::processor::types::ProcessorBatch) -> Self {
        // 将features数据转换为字节
        let mut processed_bytes = Vec::new();
        for &val in &batch.features.data {
            processed_bytes.extend_from_slice(&val.to_le_bytes());
        }
        
        let mut metadata = batch.metadata.clone();
        metadata.insert("format".to_string(), format!("{:?}", batch.format));
        if !batch.field_names.is_empty() {
            metadata.insert("fields".to_string(), batch.field_names.join(","));
        }
        // 将shape信息存储在metadata中
        let shape_str = serde_json::to_string(&batch.features.shape).unwrap_or_default();
        metadata.insert("shape".to_string(), shape_str);
        metadata.insert("data_type".to_string(), "float32".to_string());
        metadata.insert("processing_steps".to_string(), "[]".to_string());

        crate::core::interfaces::ProcessedData {
            id: batch.id,
            data: processed_bytes,
            format: "tensor".to_string(),
            size: processed_bytes.len(),
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
} 

impl TryFrom<crate::core::types::CoreDataBatch> for crate::data::exports::DataBatch {
    type Error = crate::error::Error;

    fn try_from(core: crate::core::types::CoreDataBatch) -> Result<Self, Self::Error> {
        // 提取特征与标签
        let mut data: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        let mut feature_dim = 0usize;

        // 从core.data中提取特征数据
        if !core.data.is_empty() {
            let feat = &core.data[0];
            feature_dim = feat.shape.get(1).copied().unwrap_or(0);
            if feat.shape.len() == 2 {
                let rows = feat.shape[0];
                let cols = feat.shape[1];
                for r in 0..rows {
                    let start = r.saturating_mul(cols);
                    let end = start.saturating_add(cols);
                    let row = feat.data.get(start..end)
                        .ok_or_else(|| crate::error::Error::invalid_input("features tensor shape/data mismatch".to_string()))?;
                    data.push(row.to_vec());
                }
            } else if feat.shape.len() == 1 {
                // 单向量，视为一行
                feature_dim = feat.shape[0];
                data.push(feat.data.clone());
            }
        }

        if let Some(ref lbl) = core.labels {
            // 将labels展平成一维
            // lbl 是 Vec<CoreTensorData>，需要提取每个 tensor 的 data
            for tensor in lbl {
                labels.extend_from_slice(&tensor.data);
            }
        }

        let batch_size = if core.batch_size > 0 { core.batch_size } else { data.len() };
        
        // 从metadata中获取sequence_length，如果没有则为None
        let sequence_length = core.metadata.as_ref()
            .and_then(|m| m.get("sequence_length"))
            .and_then(|s| s.parse::<usize>().ok());

        Ok(crate::data::exports::DataBatch {
            id: core.id,
            data,
            labels,
            batch_size,
            batch_index: 0,
            feature_dim,
            sequence_length,
            metadata: core.metadata.unwrap_or_default(),
            created_at: core.created_at,
        })
    }
}

impl From<crate::core::interfaces::ProcessedData> for crate::data::processor::types::ProcessorBatch {
    fn from(pd: crate::core::interfaces::ProcessedData) -> Self {
        // 从metadata中获取shape信息
        let shape = pd.metadata.get("shape")
            .and_then(|s| serde_json::from_str::<Vec<usize>>(s).ok())
            .unwrap_or_else(|| vec![pd.data.len() / 4]); // 假设是f32数据
        
        // 将字节数据转换为f32数组
        let mut f32_data = Vec::new();
        for chunk in pd.data.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32_data.push(f32::from_le_bytes(bytes));
        }
        
        let features = crate::core::types::CoreTensorData {
            id: pd.id.clone(),
            shape,
            data: f32_data,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: pd.created_at,
            updated_at: pd.updated_at,
        };
        let mut metadata = pd.metadata.clone();
        metadata.entry("source".to_string()).or_insert_with(|| "ProcessedData".to_string());
        crate::data::processor::types::ProcessorBatch {
            id: pd.id,
            features,
            labels: None,
            metadata,
            format: crate::data::DataFormat::CSV,
            field_names: Vec::new(),
            records: Vec::new(),
        }
    }
} 
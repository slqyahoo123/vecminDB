use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::error::Result;

// 重新导出DataValue
pub use crate::data::value::DataValue;

/// 数据配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataConfig {
    pub format: DataFormat,
    pub batch_size: usize,
    pub shuffle: bool,
    pub cache_enabled: bool,
    pub preprocessing_steps: Vec<String>,
    pub validation_split: f32,
    /// 数据路径
    pub path: Option<String>,
    /// 数据模式
    pub schema: Option<crate::data::schema::schema::DataSchema>,
    /// 是否跳过表头
    pub skip_header: Option<bool>,
    /// 分隔符
    pub delimiter: Option<String>,
    /// 是否验证
    pub validate: Option<bool>,
    /// 额外参数
    pub extra_params: HashMap<String, String>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            format: DataFormat::CSV,
            batch_size: 32,
            shuffle: true,
            cache_enabled: true,
            preprocessing_steps: Vec::new(),
            validation_split: 0.2,
            path: None,
            schema: None,
            skip_header: None,
            delimiter: None,
            validate: None,
            extra_params: HashMap::new(),
        }
    }
}

impl DataConfig {
    /// 添加参数
    pub fn add_parameter(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.extra_params.insert(key.into(), value.into());
    }
}

/// 数据格式
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DataFormat {
    CSV,
    JSON,
    TSV,
    Parquet,
    Arrow,
    Binary,
    Custom(String),
    Vector,
    /// Tensor格式
    Tensor,
    /// JSON格式
    Json,
    /// CSV格式
    Csv,
    /// TSV格式
    Tsv,
    /// Avro格式
    Avro,
    /// 文本格式
    Text,
    /// 自定义文本格式
    CustomText(String),
    /// 矩阵格式
    Matrix {
        /// 行数
        rows: usize,
        /// 列数
        cols: usize,
        /// 数据类型
        dtype: String,
    },
    /// 自定义二进制格式
    CustomBinary(String),
}

impl Default for DataFormat {
    fn default() -> Self {
        DataFormat::CSV
    }
}

impl From<&str> for DataFormat {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "csv" => DataFormat::CSV,
            "json" => DataFormat::JSON,
            "tsv" => DataFormat::TSV,
            "parquet" => DataFormat::Parquet,
            "arrow" => DataFormat::Arrow,
            "binary" => DataFormat::Binary,
            "vector" => DataFormat::Vector,
            "tensor" => DataFormat::Tensor,
            "avro" => DataFormat::Avro,
            "text" => DataFormat::Text,
            s if s.starts_with("custom_binary:") => {
                DataFormat::CustomBinary(s.strip_prefix("custom_binary:").unwrap_or(s).to_string())
            },
            _ => DataFormat::Custom(s.to_string()),
        }
    }
}

impl From<String> for DataFormat {
    fn from(s: String) -> Self {
        DataFormat::from(s.as_str())
    }
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFormat::CSV => write!(f, "csv"),
            DataFormat::JSON => write!(f, "json"),
            DataFormat::TSV => write!(f, "tsv"),
            DataFormat::Parquet => write!(f, "parquet"),
            DataFormat::Arrow => write!(f, "arrow"),
            DataFormat::Binary => write!(f, "binary"),
            DataFormat::Vector => write!(f, "vector"),
            DataFormat::Tensor => write!(f, "tensor"),
            DataFormat::Json => write!(f, "json"),
            DataFormat::Csv => write!(f, "csv"),
            DataFormat::Tsv => write!(f, "tsv"),
            DataFormat::Avro => write!(f, "avro"),
            DataFormat::Text => write!(f, "text"),
            DataFormat::Custom(s) => write!(f, "{}", s),
            DataFormat::CustomText(s) => write!(f, "{}", s),
            DataFormat::Matrix { rows, cols, dtype } => write!(f, "matrix({}x{}, {})", rows, cols, dtype),
        }
    }
}

/// 数据集元数据
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DatasetMetadata {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub owner: String,
    pub schema: Option<String>,
    pub properties: HashMap<String, String>,
    /// 数据集标签
    pub tags: Vec<String>,
    /// 记录数量
    pub records_count: usize,
    /// 数据大小（字节）
    pub size_bytes: u64,
}

/// 数据分割
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSplit {
    pub train: String,
    pub validation: Option<String>,
    pub test: Option<String>,
    pub ratios: Vec<f32>,
    pub shuffle: bool,
}

/// 数据状态枚举
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataStatus {
    /// 已创建
    Created,
    /// 已初始化
    Initialized,
    /// 加载中
    Loading,
    /// 已加载
    Loaded,
    /// 处理中
    Processing,
    /// 已处理
    Processed,
    /// 错误
    Error(String),
    /// 已归档
    Archived,
}

impl std::fmt::Display for DataStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataStatus::Created => write!(f, "已创建"),
            DataStatus::Initialized => write!(f, "已初始化"),
            DataStatus::Loading => write!(f, "加载中"),
            DataStatus::Loaded => write!(f, "已加载"),
            DataStatus::Processing => write!(f, "处理中"),
            DataStatus::Processed => write!(f, "已处理"),
            DataStatus::Error(msg) => write!(f, "错误: {}", msg),
            DataStatus::Archived => write!(f, "已归档"),
        }
    }
}

/// 处理步骤
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingStep {
    /// 步骤名称
    pub name: String,
    /// 步骤描述
    pub description: Option<String>,
    /// 步骤参数
    pub parameters: HashMap<String, String>,
    /// 步骤创建时间
    pub created_at: DateTime<Utc>,
}

impl ProcessingStep {
    /// 创建新的处理步骤
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            parameters: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// 设置描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// 添加参数
    pub fn with_parameter(mut self, key: &str, value: &str) -> Self {
        self.parameters.insert(key.to_string(), value.to_string());
        self
    }
}

/// 处理选项
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingOptions {
    /// 是否规范化
    pub normalize: bool,
    /// 是否增强
    pub augmentation: bool,
    /// 是否过滤异常值
    pub filter_outliers: bool,
    /// 维度降低算法
    pub dimension_reduction: Option<String>,
    /// 特征提取算法
    pub feature_extraction: Option<String>,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            normalize: true,
            augmentation: false,
            filter_outliers: true,
            dimension_reduction: None,
            feature_extraction: None,
        }
    }
}

/// 统一数据项，支持多种数据格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataItem {
    /// 向量数据
    Vector(Vec<f32>),
    /// 矩阵数据
    Matrix(Vec<Vec<f32>>),
    /// 文本数据
    Text(String),
    /// 图像数据
    Image(Vec<u8>),
    /// 标量值
    Value(f32),
    /// 布尔值
    Boolean(bool),
    /// 整数值
    Integer(i64),
    /// 空值
    Null,
}

impl DataItem {
    /// 创建向量数据项
    pub fn vector(data: Vec<f32>) -> Self {
        DataItem::Vector(data)
    }
    
    /// 创建矩阵数据项
    pub fn matrix(data: Vec<Vec<f32>>) -> Self {
        DataItem::Matrix(data)
    }
    
    /// 创建文本数据项
    pub fn text(data: String) -> Self {
        DataItem::Text(data)
    }
    
    /// 创建图像数据项
    pub fn image(data: Vec<u8>) -> Self {
        DataItem::Image(data)
    }
    
    /// 创建标量值数据项
    pub fn value(data: f32) -> Self {
        DataItem::Value(data)
    }
    
    /// 获取数据项类型
    pub fn data_type(&self) -> &'static str {
        match self {
            DataItem::Vector(_) => "vector",
            DataItem::Matrix(_) => "matrix", 
            DataItem::Text(_) => "text",
            DataItem::Image(_) => "image",
            DataItem::Value(_) => "value",
            DataItem::Boolean(_) => "boolean",
            DataItem::Integer(_) => "integer",
            DataItem::Null => "null",
        }
    }
    
    /// 检查是否为空值
    pub fn is_null(&self) -> bool {
        matches!(self, DataItem::Null)
    }
    
    /// 转换为向量（如果可能）
    pub fn as_vector(&self) -> Option<&Vec<f32>> {
        match self {
            DataItem::Vector(v) => Some(v),
            _ => None,
        }
    }
    
    /// 转换为文本（如果可能）
    pub fn as_text(&self) -> Option<&String> {
        match self {
            DataItem::Text(s) => Some(s),
            _ => None,
        }
    }
} 

/// 数据类型枚举
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    /// 图像数据
    Image,
    /// 文本数据
    Text,
    /// 表格数据
    Tabular,
    /// 音频数据
    Audio,
    /// 视频数据
    Video,
    /// 时间序列数据
    TimeSeries,
    /// 向量数据
    Vector,
    /// 自定义数据类型
    Custom(String),
}

impl Default for DataType {
    fn default() -> Self {
        DataType::Tabular
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Image => write!(f, "image"),
            DataType::Text => write!(f, "text"),
            DataType::Tabular => write!(f, "tabular"),
            DataType::Audio => write!(f, "audio"),
            DataType::Video => write!(f, "video"),
            DataType::TimeSeries => write!(f, "time_series"),
            DataType::Vector => write!(f, "vector"),
            DataType::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// 数据样本结构
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSample {
    /// 样本ID
    id: String,
    /// 特征数据
    features: HashMap<String, serde_json::Value>,
    /// 数据类型
    data_type: DataType,
    /// 元数据
    metadata: HashMap<String, String>,
    /// 创建时间
    created_at: DateTime<Utc>,
    /// 标签（可选）
    label: Option<serde_json::Value>,
    /// 原始数据（可选）
    raw_data: Option<Vec<u8>>,
}

impl DataSample {
    /// 创建新的数据样本
    pub fn new(
        id: String,
        features: HashMap<String, serde_json::Value>,
        data_type: DataType,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            id,
            features,
            data_type,
            metadata: metadata.unwrap_or_default(),
            created_at: Utc::now(),
            label: None,
            raw_data: None,
        }
    }

    /// 获取样本ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 获取特征数据
    pub fn features(&self) -> &HashMap<String, serde_json::Value> {
        &self.features
    }

    /// 获取可变特征数据
    pub fn features_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        &mut self.features
    }

    /// 获取数据类型
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// 获取元数据
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// 获取可变元数据
    pub fn metadata_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.metadata
    }

    /// 获取创建时间
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// 设置标签
    pub fn set_label(&mut self, label: serde_json::Value) {
        self.label = Some(label);
    }

    /// 获取标签
    pub fn label(&self) -> Option<&serde_json::Value> {
        self.label.as_ref()
    }

    /// 设置原始数据
    pub fn set_raw_data(&mut self, data: Vec<u8>) {
        self.raw_data = Some(data);
    }

    /// 获取原始数据
    pub fn raw_data(&self) -> Option<&Vec<u8>> {
        self.raw_data.as_ref()
    }

    /// 添加特征
    pub fn add_feature(&mut self, key: String, value: serde_json::Value) {
        self.features.insert(key, value);
    }

    /// 移除特征
    pub fn remove_feature(&mut self, key: &str) -> Option<serde_json::Value> {
        self.features.remove(key)
    }

    /// 获取特定特征
    pub fn get_feature(&self, key: &str) -> Option<&serde_json::Value> {
        self.features.get(key)
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// 移除元数据
    pub fn remove_metadata(&mut self, key: &str) -> Option<String> {
        self.metadata.remove(key)
    }

    /// 获取特定元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// 验证样本数据的完整性
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("样本ID不能为空".to_string());
        }

        if self.features.is_empty() {
            return Err("特征数据不能为空".to_string());
        }

        // 根据数据类型进行特定验证
        match self.data_type {
            DataType::Image => {
                if !self.features.contains_key("width") || !self.features.contains_key("height") {
                    return Err("图像数据必须包含宽度和高度信息".to_string());
                }
            }
            DataType::Text => {
                if !self.features.contains_key("text") {
                    return Err("文本数据必须包含文本内容".to_string());
                }
            }
            DataType::Vector => {
                if !self.features.contains_key("vector") {
                    return Err("向量数据必须包含向量内容".to_string());
                }
            }
            _ => {} // 其他类型暂不验证
        }

        Ok(())
    }

    /// 转换为JSON字符串
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// 从JSON字符串创建
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// 获取样本大小（字节）
    pub fn size_bytes(&self) -> usize {
        let mut size = self.id.len();
        
        // 计算特征数据大小
        size += self.features.iter()
            .map(|(k, v)| k.len() + v.to_string().len())
            .sum::<usize>();
        
        // 计算元数据大小
        size += self.metadata.iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();
        
        // 计算原始数据大小
        if let Some(raw_data) = &self.raw_data {
            size += raw_data.len();
        }
        
        size
    }

    /// 克隆样本但生成新ID
    pub fn clone_with_new_id(&self, new_id: String) -> Self {
        let mut clone = self.clone();
        clone.id = new_id;
        clone.created_at = Utc::now();
        clone
    }
}

impl Default for DataSample {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            features: HashMap::new(),
            data_type: DataType::default(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            label: None,
            raw_data: None,
        }
    }
} 
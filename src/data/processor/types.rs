// Data Processor Types
// 数据处理器类型定义

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
// 移除循环依赖：不再直接导入 model::tensor::TensorData
// use crate::model::tensor::TensorData;
// 改为使用 core 模块的统一类型
use crate::core::CoreTensorData;
use crate::data::DataFormat;
use crate::{Error, Result};
// 引入data/schema模块
use crate::data::schema::{DataSchema, Schema as BaseSchema};
use crate::data::schema::schema::{
    FieldType as SchemaFieldType,
    FieldDefinition as SchemaFieldDefinition,
};

// 为了向后兼容，创建类型别名
pub type TensorData = CoreTensorData;
pub type ProcessedBatch = ProcessorBatch;

/// 处理后的数据批次
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorBatch {
    pub id: String,
    pub features: TensorData,
    pub labels: Option<TensorData>,
    pub metadata: HashMap<String, String>,
    pub format: DataFormat,
    /// 字段名称列表
    pub field_names: Vec<String>,
    /// 记录数据
    pub records: Vec<HashMap<String, crate::data::DataValue>>,
}

impl ProcessorBatch {
    pub fn new(
        id: String,
        features: TensorData,
        labels: Option<TensorData>,
        metadata: HashMap<String, String>,
        format: DataFormat
    ) -> Self {
        Self {
            id,
            features,
            labels,
            metadata,
            format,
            field_names: Vec::new(),
            records: Vec::new(),
        }
    }
    
    pub fn from_batch(batch: &crate::data::DataBatch) -> crate::Result<Self> {
        // 获取ID或生成一个新的（统一使用向量模块ID生成器）
        let id = batch
            .id
            .clone()
            .unwrap_or_else(|| crate::vector::generate_uuid());

        // 从 DataBatch.features 中获取特征矩阵
        let feature_matrix = batch
            .features
            .as_ref()
            .ok_or_else(|| Error::invalid_input("数据批次缺少特征数据(features)".to_string()))?;

        // 验证数据结构
        if feature_matrix.is_empty() {
            return Err(Error::invalid_input("数据批次不能为空".to_string()));
        }

        // 将特征数据转换为TensorData
        let features = Self::convert_to_tensor_data(feature_matrix)?;

        // 当前 DataBatch 中的标签数据以原始字节形式存储，转换为 TensorData 需要额外的语义信息。
        // 在这里先不将 labels 转换为 TensorData，保持为 None，并通过元数据记录提示。
        let labels = None;

        // 处理元数据（DataBatch.metadata 现在是 HashMap<String, String>）
        let mut metadata = HashMap::new();
        metadata.extend(batch.metadata.clone());

        // 添加处理时间戳（标准化时间戳与人类可读时间）
        let now: DateTime<Utc> = Utc::now();
        metadata.insert("processed_at".to_string(), now.timestamp().to_string());
        metadata.insert("processed_at_iso".to_string(), now.to_rfc3339());

        // 添加特征维度信息
        if let Some(first_row) = feature_matrix.first() {
            metadata.insert("feature_dim".to_string(), first_row.len().to_string());
        }

        // 添加批次大小信息
        metadata.insert("batch_size".to_string(), feature_matrix.len().to_string());

        // 使用批次中的格式
        let format = batch.format;

        Ok(Self {
            id,
            features,
            labels,
            metadata,
            format,
            field_names: Vec::new(),
            records: Vec::new(),
        })
    }
    
    /// 从 exports::DataBatch 创建 ProcessorBatch（兼容旧接口）
    pub fn from_data_batch(batch: crate::data::exports::DataBatch) -> crate::Result<Self> {
        use crate::data::batch::DataBatch as InternalBatch;

        // 1. 准备特征和标签矩阵
        let features: Vec<Vec<f32>> = batch.data.clone();

        // 将一维标签 Vec<f32> 包装成二维矩阵 (N x 1)，以便复用 convert_to_tensor_data
        let labels_matrix: Option<Vec<Vec<f32>>> = if !batch.labels.is_empty() {
            Some(batch.labels.iter().map(|v| vec![*v]).collect())
        } else {
            None
        };

        // 2. 构建内部 DataBatch，利用其已有的字段和构造函数
        let mut internal = InternalBatch::new("", 0, features.len());
        // 设置批次 ID、元数据、特征
        internal.id = Some(batch.id.clone());
        internal.metadata = batch.metadata.clone();
        internal = internal.with_features(features);

        // 当前导出批次不包含原始 DataValue 形式的记录，这里不填充 records/labels，
        // 仅通过 features/labels_matrix 构建张量数据。

        // 3. 使用 from_batch 生成 ProcessorBatch
        let mut processor_batch = ProcessorBatch::from_batch(&internal)?;

        // 如果有标签，则转换为 TensorData 填充到 ProcessorBatch.labels
        if let Some(label_mat) = labels_matrix {
            let label_tensor = Self::convert_to_tensor_data(&label_mat)?;
            processor_batch.labels = Some(label_tensor);
        }

        // 补充一些常用元数据
        processor_batch
            .metadata
            .insert("source_batch_id".to_string(), batch.id);
        processor_batch
            .metadata
            .insert("batch_size".to_string(), internal.batch_size.to_string());

        Ok(processor_batch)
    }
    
    // 辅助方法：将Vec<Vec<f32>>转换为TensorData
    fn convert_to_tensor_data(data: &Vec<Vec<f32>>) -> Result<TensorData> {
        if data.is_empty() {
            return Ok(TensorData::empty());
        }
        
        // 确保所有行的长度一致
        let first_row_len = data[0].len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != first_row_len {
                return Err(Error::invalid_input(
                    format!("数据行长度不一致: 第一行长度 {}, 第 {} 行长度 {}", 
                            first_row_len, i, row.len())
                ));
            }
        }
        
        // 创建TensorData
        let shape = vec![data.len(), first_row_len];
        
        // 将二维数据展平为一维
        let mut flat_data = Vec::with_capacity(data.len() * first_row_len);
        for row in data {
            flat_data.extend_from_slice(row);
        }
        
        Ok(TensorData::new(shape, flat_data))
    }
}

/// 数据类型枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// 字符串类型
    String,
    /// 整数类型
    Integer,
    /// 浮点数类型
    Float,
    /// 布尔类型
    Boolean,
    /// JSON类型
    Json,
    /// CSV类型
    Csv,
    /// 文本类型
    Text,
    /// 二进制类型
    Binary,
    /// 空值类型
    Null,
    /// 数值类型（通用）
    Numeric,
    /// 分类类型
    Categorical,
    /// 日期时间类型
    DateTime,
    /// 数组类型
    Array(Box<DataType>),
    /// 对象类型
    Object,
    /// 任意类型
    Any,
    /// 未知类型
    Unknown,
    /// 向量类型
    Vector,
}

/// 列信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: DataType,
    pub is_nullable: bool,
    pub validation_rules: Vec<ValidationRule>,
}

/// 验证规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    NotNull,
    Range { min: f64, max: f64 },
    Length { min: usize, max: usize },
    Pattern(String),
    Custom(String),
}

/// 处理后的数据集
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorDataset {
    pub id: String,
    pub features: TensorData,
    pub labels: Option<TensorData>,
    pub metadata: HashMap<String, String>,
    pub format: DataFormat,
}

impl ProcessorDataset {
    pub fn new(id: String) -> Self {
        Self {
            id,
            features: TensorData::empty(),
            labels: None,
            metadata: HashMap::new(),
            format: DataFormat::CSV,
        }
    }
    
    pub fn from_batch(batch: &crate::data::DataBatch) -> crate::Result<Self> {
        // 使用ProcessorBatch的转换逻辑，然后转为ProcessorDataset
        let processed_batch = ProcessorBatch::from_batch(batch)?;
        
        // 构建ProcessedDataset
        let dataset = Self {
            id: processed_batch.id,
            features: processed_batch.features,
            labels: processed_batch.labels,
            metadata: processed_batch.metadata,
            format: processed_batch.format,
        };
        
        // 添加数据集特定元数据
        let mut metadata = dataset.metadata.clone();
        metadata.insert("dataset_type".to_string(), "processed".to_string());
        metadata.insert("created_at".to_string(), chrono::Utc::now().timestamp().to_string());
        
        Ok(Self {
            metadata,
            ..dataset
        })
    }
    
    // 从多个批次合并成一个数据集
    pub fn from_batches(batches: &[crate::data::DataBatch]) -> crate::Result<Self> {
        if batches.is_empty() {
            return Err(Error::invalid_input("无法从空的批次列表创建数据集".to_string()));
        }
        
        // 处理第一个批次以获取基本结构
        let first_processed = Self::from_batch(&batches[0])?;
        
        // 如果只有一个批次，直接返回
        if batches.len() == 1 {
            return Ok(first_processed);
        }
        
        // 收集所有批次的特征和标签
        let mut all_features = Vec::new();
        let mut all_labels = Vec::new();
        let mut total_rows = 0;
        
        for batch in batches {
            let processed = ProcessedBatch::from_batch(batch)?;
            
            // 合并特征
            let batch_features = &processed.features.data;
            all_features.extend_from_slice(batch_features);
            
            // 合并标签(如果存在)
            if let Some(labels) = &processed.labels {
                let batch_labels = &labels.data;
                all_labels.extend_from_slice(batch_labels);
            }
            
            // 根据张量形状累加样本数（shape[0] 为样本数）
            let rows = processed
                .features
                .shape
                .get(0)
                .cloned()
                .unwrap_or(0);
            total_rows += rows;
        }
        
        // 计算合并后的形状
        let feature_dim = if let Some(first_row) = batches
            .get(0)
            .and_then(|b| b.features.as_ref())
            .and_then(|f| f.first())
        {
            first_row.len()
        } else {
            0
        };
        
        let features_shape = vec![total_rows, feature_dim];
        let features = TensorData::new(features_shape, all_features);
        
        // 处理标签(如果所有批次都有标签)
        let labels = if all_labels.len() == total_rows * feature_dim {
            Some(TensorData::new(features_shape.clone(), all_labels))
        } else if !all_labels.is_empty() {
            // 标签数量不一致，可能存在错误
            return Err(Error::invalid_state("批次标签维度不一致".to_string()));
        } else {
            None
        };
        
        // 使用新的UUID作为合并数据集的ID
        let id = uuid::Uuid::new_v4().to_string();
        
        // 创建合并后的元数据
        let mut metadata = first_processed.metadata.clone();
        metadata.insert("dataset_type".to_string(), "merged".to_string());
        metadata.insert("batch_count".to_string(), batches.len().to_string());
        metadata.insert("total_rows".to_string(), total_rows.to_string());
        metadata.insert("created_at".to_string(), chrono::Utc::now().timestamp().to_string());
        
        Ok(Self {
            id,
            features,
            labels,
            metadata,
            format: first_processed.format,
        })
    }
}

/// 处理器选项
/// Processor options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessorOptions {
    /// 基本选项
    /// Basic options
    pub options: HashMap<String, String>,
    
    /// 处理器类型
    /// Processor type
    pub processor_type: Option<String>,
    
    /// 是否启用缓存
    /// Whether to enable cache
    pub enable_cache: bool,
    
    /// 最大缓存大小
    /// Maximum cache size
    pub max_cache_size: Option<usize>,
}

impl ProcessorOptions {
    /// 创建新的处理器选项
    /// Create new processor options
    pub fn new() -> Self {
        Self {
            options: HashMap::new(),
            processor_type: None,
            enable_cache: true,
            max_cache_size: None,
        }
    }
    
    /// 设置基本选项
    /// Set basic option
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
    
    /// 设置多个基本选项
    /// Set multiple basic options
    pub fn with_options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }
    
    /// 设置处理器类型
    /// Set processor type
    pub fn with_processor_type(mut self, processor_type: impl Into<String>) -> Self {
        self.processor_type = Some(processor_type.into());
        self
    }
    
    /// 设置是否启用缓存
    /// Set whether to enable cache
    pub fn with_enable_cache(mut self, enable_cache: bool) -> Self {
        self.enable_cache = enable_cache;
        self
    }
    
    /// 设置最大缓存大小
    /// Set maximum cache size
    pub fn with_max_cache_size(mut self, max_cache_size: usize) -> Self {
        self.max_cache_size = Some(max_cache_size);
        self
    }
    
    /// 获取选项值
    /// Get option value
    pub fn get(&self, key: &str) -> Option<&String> {
        self.options.get(key)
    }
    
    /// 获取选项值，如果不存在则返回默认值
    /// Get option value, return default value if not exists
    pub fn get_or(&self, key: &str, default_value: &str) -> String {
        self.options.get(key).map(|v| v.clone()).unwrap_or_else(|| default_value.to_string())
    }
    
    /// 获取布尔选项值
    /// Get boolean option value
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.options.get(key).and_then(|v| v.parse::<bool>().ok())
    }
    
    /// 获取布尔选项值，如果不存在则返回默认值
    /// Get boolean option value, return default value if not exists
    pub fn get_bool_or(&self, key: &str, default_value: bool) -> bool {
        self.get_bool(key).unwrap_or(default_value)
    }
    
    /// 获取整数选项值
    /// Get integer option value
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.options.get(key).and_then(|v| v.parse::<i64>().ok())
    }
    
    /// 获取整数选项值，如果不存在则返回默认值
    /// Get integer option value, return default value if not exists
    pub fn get_int_or(&self, key: &str, default_value: i64) -> i64 {
        self.get_int(key).unwrap_or(default_value)
    }
    
    /// 获取浮点数选项值
    /// Get float option value
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.options.get(key).and_then(|v| v.parse::<f64>().ok())
    }
    
    /// 获取浮点数选项值，如果不存在则返回默认值
    /// Get float option value, return default value if not exists
    pub fn get_float_or(&self, key: &str, default_value: f64) -> f64 {
        self.get_float(key).unwrap_or(default_value)
    }
    
    /// 获取字符串选项值（返回克隆的字符串）
    /// Get string option value (returns cloned string)
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.get(key).map(|s| s.clone())
    }
    
    /// 获取整数选项值（返回usize）
    /// Get integer option value (returns usize)
    pub fn get_integer(&self, key: &str) -> Option<usize> {
        self.get_int(key).map(|i| i as usize)
    }
    
    /// 获取自定义选项（返回所有选项的克隆）
    /// Get custom options (returns cloned options)
    pub fn get_custom_options(&self) -> Option<HashMap<String, String>> {
        if self.options.is_empty() {
            None
        } else {
            Some(self.options.clone())
        }
    }
    
    /// 从选项映射创建
    /// Create from options map
    pub fn from_map(map: HashMap<String, String>) -> Self {
        let mut options = Self::new();
        
        if let Some(processor_type) = map.get("processor_type") {
            options.processor_type = Some(processor_type.clone());
        }
        
        if let Some(enable_cache) = map.get("enable_cache") {
            if let Ok(value) = enable_cache.parse::<bool>() {
                options.enable_cache = value;
            }
        }
        
        if let Some(max_cache_size) = map.get("max_cache_size") {
            if let Ok(value) = max_cache_size.parse::<usize>() {
                options.max_cache_size = Some(value);
            }
        }
        
        // 添加所有其他选项
        for (key, value) in map {
            if key != "processor_type" && key != "enable_cache" && key != "max_cache_size" {
                options.options.insert(key, value);
            }
        }
        
        options
    }
    
    /// 转换为选项映射
    /// Convert to options map
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = self.options.clone();
        
        if let Some(processor_type) = &self.processor_type {
            map.insert("processor_type".to_string(), processor_type.clone());
        }
        
        map.insert("enable_cache".to_string(), self.enable_cache.to_string());
        
        if let Some(max_cache_size) = self.max_cache_size {
            map.insert("max_cache_size".to_string(), max_cache_size.to_string());
        }
        
        map
    }
}

/// 处理器专用的Schema适配器，用于将数据处理相关的功能添加到标准Schema上
/// Processor-specific Schema adapter that adds data processing functionality to the standard Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorSchema {
    /// 底层Schema实现
    pub schema: BaseSchema,
    /// 处理器特定的元数据
    pub processor_metadata: HashMap<String, String>,
}

impl ProcessorSchema {
    /// 创建新的处理器Schema
    pub fn new(schema: BaseSchema) -> Self {
        Self {
            schema,
            processor_metadata: HashMap::new(),
        }
    }
    
    /// 从DataSchema创建处理器Schema
    pub fn from_data_schema(data_schema: DataSchema) -> Self {
        Self {
            schema: data_schema,
            processor_metadata: HashMap::new(),
        }
    }
    
    /// 设置处理器元数据
    pub fn with_processor_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.processor_metadata.insert(key.into(), value.into());
        self
    }
    
    /// 转换ProcessorSchema为DataType类型的映射
    pub fn to_data_type_map(&self) -> HashMap<String, DataType> {
        let mut type_map = HashMap::new();
        
        for field in &self.schema.fields {
            let data_type = match &field.field_type {
                SchemaFieldType::Numeric => DataType::Numeric,
                SchemaFieldType::Categorical => DataType::Categorical,
                SchemaFieldType::Text => DataType::Text,
                SchemaFieldType::DateTime => DataType::DateTime,
                SchemaFieldType::Boolean => DataType::Boolean,
                SchemaFieldType::Array(_) => DataType::Array(Box::new(DataType::Numeric)),
                SchemaFieldType::Object(_) => DataType::Object,
                SchemaFieldType::Custom(_) => DataType::Unknown,
                SchemaFieldType::Image => DataType::Unknown,
                SchemaFieldType::Audio => DataType::Unknown,
                SchemaFieldType::Video => DataType::Unknown,
                _ => DataType::Unknown,
            };
            
            type_map.insert(field.name.clone(), data_type);
        }
        
        type_map
    }
    
    /// 从CSV转换为DataSchema，然后创建ProcessorSchema
    pub fn from_csv_header(header: &[String]) -> Self {
        let mut fields = Vec::new();
        
        for field_name in header {
            fields.push(SchemaFieldDefinition {
                name: field_name.clone(),
                field_type: SchemaFieldType::Text, // 默认为文本类型
                data_type: None,
                required: false,
                nullable: false,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            });
        }
        
        let schema = DataSchema::new_with_fields(fields, "csv_schema", "1.0");
        Self::from_data_schema(schema)
    }
    
    /// 转换为JSON
    pub fn to_json(&self) -> Result<String> {
        let mut schema_value = serde_json::to_value(&self.schema)?;
        
        if let serde_json::Value::Object(ref mut obj) = schema_value {
            let metadata = serde_json::to_value(&self.processor_metadata)?;
            obj.insert("processor_metadata".to_string(), metadata);
        }
        
        serde_json::to_string_pretty(&schema_value)
            .map_err(|e| Error::serialization(format!("无法序列化ProcessorSchema: {}", e)))
    }
    
    /// 从JSON创建
    pub fn from_json(json: &str) -> Result<Self> {
        let mut schema: BaseSchema = serde_json::from_str(json)?;
        let mut processor_metadata = HashMap::new();
        
        let value: serde_json::Value = serde_json::from_str(json)?;
        if let serde_json::Value::Object(obj) = value {
            if let Some(serde_json::Value::Object(metadata)) = obj.get("processor_metadata") {
                for (k, v) in metadata {
                    if let serde_json::Value::String(s) = v {
                        processor_metadata.insert(k.clone(), s.clone());
                    }
                }
            }
        }
        
        Ok(Self {
            schema,
            processor_metadata,
        })
    }
}

/// 兼容原processor/types中Schema字段和结构的SimpleSchema定义
/// 这是为了兼容现有的代码，特别是core.rs中schema模块
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimpleSchema {
    /// 字段类型映射
    pub fields: HashMap<String, DataType>,
    /// 主键字段
    pub primary_key: Option<String>,
}

impl SimpleSchema {
    /// 创建新的空Schema
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            primary_key: None,
        }
    }
    
    /// 从HashMap创建Schema
    pub fn from_map(fields: HashMap<String, DataType>, primary_key: Option<String>) -> Self {
        Self {
            fields,
            primary_key,
        }
    }
    
    /// 从ProcessorSchema转换
    pub fn from_processor_schema(schema: &ProcessorSchema) -> Self {
        Self {
            fields: schema.to_data_type_map(),
            primary_key: schema.schema.primary_key.as_ref().map(|keys| keys[0].clone()),
        }
    }
    
    /// 转换为ProcessorSchema
    pub fn to_processor_schema(&self) -> ProcessorSchema {
        let mut fields = Vec::new();
        
        for (name, data_type) in &self.fields {
            let field_type = match data_type {
                DataType::Numeric => SchemaFieldType::Numeric,
                DataType::Categorical => SchemaFieldType::Categorical,
                DataType::DateTime => SchemaFieldType::DateTime,
                DataType::Text => SchemaFieldType::Text,
                DataType::Boolean => SchemaFieldType::Boolean,
                DataType::Array(_) => SchemaFieldType::Array(Box::new(SchemaFieldType::Numeric)),
                DataType::Vector => SchemaFieldType::Array(Box::new(SchemaFieldType::Numeric)),
                DataType::Object => SchemaFieldType::Object(HashMap::new()),
                _ => SchemaFieldType::Text, // 默认为文本
            };
            
            fields.push(SchemaFieldDefinition {
                name: name.clone(),
                field_type,
                data_type: None,
                required: false,
                nullable: false,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            });
        }
        
        let mut schema = DataSchema::new("processor_schema", "1.0");
        if let Err(e) = schema.add_fields(fields) {
            // 如果添加字段失败，记录错误但不中断流程
            log::error!("添加字段到Schema时出错: {}", e);
        }
        
        if let Some(pk) = &self.primary_key {
            if let Err(e) = schema.set_primary_key(vec![pk.clone()]) {
                log::error!("设置主键时出错: {}", e);
            }
        }
        
        ProcessorSchema::from_data_schema(schema)
    }
    
    /// 从CSV头创建
    pub fn from_csv_header(header: &[String], _infer_types: bool) -> Self {
        let mut fields = HashMap::new();
        
        for field_name in header {
            fields.insert(field_name.clone(), DataType::Text);
        }
        
        if !header.is_empty() {
            Self {
                fields,
                primary_key: Some(header[0].clone()),
            }
        } else {
            Self {
                fields,
                primary_key: None,
            }
        }
    }
    
    /// 从JSON创建
    pub fn from_json(json: &str) -> Result<Self> {
        let processor_schema = ProcessorSchema::from_json(json)?;
        Ok(Self::from_processor_schema(&processor_schema))
    }
}

/// 这是一个通用适配器，能够根据需要返回不同类型的Schema
/// 对外暴露为Schema类型，保持与原有代码的兼容性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaAdapter {
    /// 简单Schema，对应原有processor/types中的Schema
    Simple(SimpleSchema),
    /// 处理器Schema，使用data/schema定义
    Processor(ProcessorSchema),
}

impl Default for SchemaAdapter {
    fn default() -> Self {
        Self::Simple(SimpleSchema::new())
    }
}

impl SchemaAdapter {
    /// 创建新的空适配器
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 从简单Schema创建
    pub fn from_simple(schema: SimpleSchema) -> Self {
        Self::Simple(schema)
    }
    
    /// 从处理器Schema创建
    pub fn from_processor(schema: ProcessorSchema) -> Self {
        Self::Processor(schema)
    }
    
    /// 从CSV头创建
    pub fn from_csv_header(header: &[String], infer_types: bool) -> Self {
        Self::Simple(SimpleSchema::from_csv_header(header, infer_types))
    }
    
    /// 从JSON创建
    pub fn from_json(json: &str) -> Result<Self> {
        let simple = SimpleSchema::from_json(json)?;
        Ok(Self::Simple(simple))
    }
    
    /// 获取字段类型映射
    pub fn fields(&self) -> HashMap<String, DataType> {
        match self {
            Self::Simple(s) => s.fields.clone(),
            Self::Processor(p) => p.to_data_type_map(),
        }
    }
    
    /// 获取主键
    pub fn primary_key(&self) -> Option<String> {
        match self {
            Self::Simple(s) => s.primary_key.clone(),
            Self::Processor(p) => p.schema.primary_key.as_ref().and_then(|keys| keys.first().cloned()),
        }
    }
    
    /// 转换为ProcessorSchema
    pub fn to_processor_schema(&self) -> ProcessorSchema {
        match self {
            Self::Simple(s) => s.to_processor_schema(),
            Self::Processor(p) => p.clone(),
        }
    }
    
    /// 转换为SimpleSchema
    pub fn to_simple_schema(&self) -> SimpleSchema {
        match self {
            Self::Simple(s) => s.clone(),
            Self::Processor(p) => SimpleSchema::from_processor_schema(p),
        }
    }
    
    /// 验证Schema
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Simple(_) => Ok(()),  // 简单Schema不需要验证
            Self::Processor(_) => Ok(()), // ProcessorSchema 无数据上下文，跳过校验
        }
    }
}

// 为兼容现有代码，提供Schema类型别名
pub type Schema = SchemaAdapter;

/// 数据上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataContext {
    /// 上下文ID
    pub id: String,
    /// 数据源信息
    pub source: String,
    /// 处理时间戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 处理状态
    pub status: ProcessingStatus,
}

impl DataContext {
    pub fn new(id: String, source: String) -> Self {
        Self {
            id,
            source,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
            status: ProcessingStatus::Pending,
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// 处理状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// 等待处理
    Pending,
    /// 处理中
    Processing,
    /// 已完成
    Completed,
    /// 失败
    Failed,
}

/// 标准特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardFeatureConfig {
    /// 特征维度
    pub feature_dim: usize,
    /// 标准化配置
    pub normalization: NormalizationConfig,
    /// 编码配置
    pub encoding: EncodingConfig,
    /// 自定义配置
    pub custom_config: HashMap<String, String>,
}

impl StandardFeatureConfig {
    pub fn new(feature_dim: usize) -> Self {
        Self {
            feature_dim,
            normalization: NormalizationConfig::default(),
            encoding: EncodingConfig::default(),
            custom_config: HashMap::new(),
        }
    }
}

/// 标准化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// 是否启用标准化
    pub enabled: bool,
    /// 标准化方法
    pub method: NormalizationMethod,
    /// 自定义参数
    pub parameters: HashMap<String, f64>,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: NormalizationMethod::StandardScore,
            parameters: HashMap::new(),
        }
    }
}

/// 标准化方法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// 标准分数标准化 (z-score)
    StandardScore,
    /// 最小-最大标准化
    MinMax,
    /// 单位向量标准化
    UnitVector,
}

/// 编码配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// 是否启用编码
    pub enabled: bool,
    /// 编码方法
    pub method: EncodingMethod,
    /// 自定义参数
    pub parameters: HashMap<String, String>,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: EncodingMethod::OneHot,
            parameters: HashMap::new(),
        }
    }
}

/// 编码方法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingMethod {
    /// 独热编码
    OneHot,
    /// 标签编码
    Label,
    /// 序数编码
    Ordinal,
}

/// 处理错误
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("数据格式错误: {0}")]
    DataFormat(String),
    
    #[error("验证失败: {0}")]
    Validation(String),
    
    #[error("转换错误: {0}")]
    Transformation(String),
    
    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("序列化错误: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("其他错误: {0}")]
    Other(String),
}

/// 处理器类型
pub use super::types_core::ProcessorType;

/// 处理器元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorMetadata {
    /// 处理器名称
    pub name: String,
    /// 处理器版本
    pub version: String,
    /// 处理器描述
    pub description: Option<String>,
    /// 处理器类型
    pub processor_type: ProcessorType,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 更新时间
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// 自定义元数据
    pub custom_metadata: HashMap<String, String>,
}

impl ProcessorMetadata {
    pub fn new(name: String, version: String, processor_type: ProcessorType) -> Self {
        let now = chrono::Utc::now();
        Self {
            name,
            version,
            description: None,
            processor_type,
            created_at: now,
            updated_at: now,
            custom_metadata: HashMap::new(),
        }
    }
}

/// 核心处理模块
pub mod core {
    use super::*;
    use crate::data::DataBatch;
    
    /// 数据管道
    #[derive(Debug, Clone)]
    pub struct DataPipeline {
        /// 管道ID
        pub id: String,
        /// 管道名称
        pub name: String,
        /// 处理步骤
        pub steps: Vec<ProcessingStep>,
        /// 管道配置
        pub config: PipelineConfig,
    }
    
    impl DataPipeline {
        pub fn new(id: String, name: String) -> Self {
            Self {
                id,
                name,
                steps: Vec::new(),
                config: PipelineConfig::default(),
            }
        }
        
        pub fn add_step(&mut self, step: ProcessingStep) {
            self.steps.push(step);
        }
        
        pub fn process(&self, batch: DataBatch) -> Result<DataBatch> {
            let mut result = batch;
            for step in &self.steps {
                result = step.process(result)?;
            }
            Ok(result)
        }
    }
    
    /// 处理步骤
    #[derive(Debug, Clone)]
    pub struct ProcessingStep {
        /// 步骤ID
        pub id: String,
        /// 步骤名称
        pub name: String,
        /// 步骤类型
        pub step_type: String,
        /// 步骤配置
        pub config: HashMap<String, String>,
    }
    
    impl ProcessingStep {
        pub fn new(id: String, name: String, step_type: String) -> Self {
            Self {
                id,
                name,
                step_type,
                config: HashMap::new(),
            }
        }
        
        pub fn process(&self, batch: DataBatch) -> Result<DataBatch> {
            // 简单的处理实现，实际中应该根据step_type进行不同的处理
            match self.step_type.as_str() {
                "normalize" => self.normalize_batch(batch),
                "filter" => self.filter_batch(batch),
                "transform" => self.transform_batch(batch),
                _ => Ok(batch),
            }
        }
        
        fn normalize_batch(&self, mut batch: DataBatch) -> Result<DataBatch> {
            // 实现标准化逻辑：对 features 中每一行 Vec<f32> 做标准化
            if let Some(ref mut features) = batch.features {
                for feature_row in features.iter_mut() {
                    if feature_row.is_empty() {
                        continue;
                    }
                    let sum: f32 = feature_row.iter().copied().sum();
                    let mean = sum / feature_row.len() as f32;
                    
                    let variance: f32 = feature_row
                        .iter()
                        .map(|x| {
                            let diff = *x - mean;
                            diff * diff
                        })
                        .sum::<f32>()
                        / feature_row.len() as f32;
                    let std_dev = variance.sqrt();
                    
                    if std_dev > 0.0 {
                        for value in feature_row.iter_mut() {
                            *value = (*value - mean) / std_dev;
                        }
                    }
                }
            }
            Ok(batch)
        }
        
        fn filter_batch(&self, mut batch: DataBatch) -> Result<DataBatch> {
            // 实现过滤逻辑，例如移除异常值
            if let Some(threshold_str) = self.config.get("threshold") {
                if let Ok(threshold) = threshold_str.parse::<f32>() {
                    if let Some(ref mut features) = batch.features {
                        features.retain(|row| row.iter().all(|&val| val.abs() <= threshold));
                        // 同时更新标签
                        if let Some(ref mut labels) = batch.labels {
                            labels.truncate(features.len());
                        }
                    }
                }
            }
            Ok(batch)
        }
        
        fn transform_batch(&self, mut batch: DataBatch) -> Result<DataBatch> {
            // 实现变换逻辑，例如应用函数变换
            if let Some(transform_type) = self.config.get("type") {
                if let Some(ref mut features) = batch.features {
                    match transform_type.as_str() {
                        "log" => {
                            for feature_row in features.iter_mut() {
                                for value in feature_row.iter_mut() {
                                    if *value > 0.0 {
                                        *value = value.ln();
                                    }
                                }
                            }
                        },
                        "sqrt" => {
                            for feature_row in features.iter_mut() {
                                for value in feature_row.iter_mut() {
                                    if *value >= 0.0 {
                                        *value = value.sqrt();
                                    }
                                }
                            }
                        },
                        _ => {}
                    }
                }
            }
            Ok(batch)
        }
    }
    
    /// 管道配置
    #[derive(Debug, Clone)]
    pub struct PipelineConfig {
        /// 并行度
        pub parallelism: usize,
        /// 批次大小
        pub batch_size: usize,
        /// 超时时间(毫秒)
        pub timeout_ms: u64,
        /// 自定义配置
        pub custom_config: HashMap<String, String>,
    }
    
    impl Default for PipelineConfig {
        fn default() -> Self {
            Self {
                parallelism: 1,
                batch_size: 1000,
                timeout_ms: 30000,
                custom_config: HashMap::new(),
            }
        }
    }
    
    /// 处理统计
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProcessingStats {
        /// 处理的记录数
        pub records_processed: usize,
        /// 处理的批次数
        pub batches_processed: usize,
        /// 处理时间(毫秒)
        pub processing_time_ms: u64,
        /// 错误数
        pub error_count: usize,
        /// 警告数
        pub warning_count: usize,
        /// 吞吐量(记录/秒)
        pub throughput: f64,
    }
    
    impl ProcessingStats {
        pub fn new() -> Self {
            Self {
                records_processed: 0,
                batches_processed: 0,
                processing_time_ms: 0,
                error_count: 0,
                warning_count: 0,
                throughput: 0.0,
            }
        }
        
        pub fn calculate_throughput(&mut self) {
            if self.processing_time_ms > 0 {
                self.throughput = (self.records_processed as f64 * 1000.0) / self.processing_time_ms as f64;
            }
        }
    }
    
    /// 数据质量指标
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DataQualityMetrics {
        /// 完整性评分(0-100)
        pub completeness_score: f64,
        /// 准确性评分(0-100)
        pub accuracy_score: f64,
        /// 一致性评分(0-100)
        pub consistency_score: f64,
        /// 及时性评分(0-100)
        pub timeliness_score: f64,
        /// 总体质量评分(0-100)
        pub overall_score: f64,
        /// 质量问题
        pub quality_issues: Vec<QualityIssue>,
    }
    
    impl DataQualityMetrics {
        pub fn new() -> Self {
            Self {
                completeness_score: 0.0,
                accuracy_score: 0.0,
                consistency_score: 0.0,
                timeliness_score: 0.0,
                overall_score: 0.0,
                quality_issues: Vec::new(),
            }
        }
        
        pub fn calculate_overall_score(&mut self) {
            self.overall_score = (self.completeness_score + self.accuracy_score + 
                                self.consistency_score + self.timeliness_score) / 4.0;
        }
    }
    
    /// 质量问题
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QualityIssue {
        /// 问题类型
        pub issue_type: String,
        /// 问题描述
        pub description: String,
        /// 严重程度
        pub severity: Severity,
        /// 影响的字段
        pub affected_fields: Vec<String>,
        /// 建议修复方案
        pub suggested_fix: Option<String>,
    }
    
    /// 严重程度
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Severity {
        Low,
        Medium,
        High,
        Critical,
    }
    
    /// 批处理器
    #[derive(Debug, Clone)]
    pub struct BatchProcessor {
        /// 处理器配置
        pub config: BatchConfig,
        /// 处理统计
        pub stats: ProcessingStats,
    }
    
    impl BatchProcessor {
        pub fn new(config: BatchConfig) -> Self {
            Self {
                config,
                stats: ProcessingStats::new(),
            }
        }
        
        pub fn process_batch(&mut self, batch: DataBatch) -> Result<ProcessorBatch> {
            let start_time = std::time::Instant::now();
            
            // 处理批次
            let processed = ProcessorBatch::from_batch(&batch)?;
            
            // 更新统计
            if let Some(ref features) = batch.features {
                self.stats.records_processed += features.len();
            }
            self.stats.batches_processed += 1;
            self.stats.processing_time_ms += start_time.elapsed().as_millis() as u64;
            self.stats.calculate_throughput();
            
            Ok(processed)
        }
    }
    
    /// 批处理配置
    #[derive(Debug, Clone)]
    pub struct BatchConfig {
        /// 批次大小
        pub batch_size: usize,
        /// 最大内存使用(MB)
        pub max_memory_mb: usize,
        /// 并行工作线程数
        pub worker_threads: usize,
        /// 处理超时(秒)
        pub timeout_seconds: u64,
    }
    
    impl Default for BatchConfig {
        fn default() -> Self {
            Self {
                batch_size: 1000,
                max_memory_mb: 512,
                worker_threads: 4,
                timeout_seconds: 300,
            }
        }
    }
    
    /// 高级处理器
    #[derive(Debug, Clone)]
    pub struct AdvancedProcessor {
        /// 高级配置
        pub config: AdvancedProcessingConfig,
        /// 数据管道
        pub pipeline: Option<DataPipeline>,
        /// 质量指标
        pub quality_metrics: DataQualityMetrics,
    }
    
    impl AdvancedProcessor {
        pub fn new(config: AdvancedProcessingConfig) -> Self {
            Self {
                config,
                pipeline: None,
                quality_metrics: DataQualityMetrics::new(),
            }
        }
        
        pub fn with_pipeline(mut self, pipeline: DataPipeline) -> Self {
            self.pipeline = Some(pipeline);
            self
        }
        
        pub fn process_advanced(&mut self, batch: DataBatch) -> Result<ProcessorBatch> {
            // 如果有管道，先通过管道处理
            let processed_batch = if let Some(ref pipeline) = self.pipeline {
                pipeline.process(batch)?
            } else {
                batch
            };
            
            // 应用高级处理逻辑
            let result = ProcessorBatch::from_batch(&processed_batch)?;
            
            // 更新质量指标
            self.update_quality_metrics(&processed_batch);
            
            Ok(result)
        }
        
        fn update_quality_metrics(&mut self, batch: &DataBatch) {
            // 计算完整性评分
            if let Some(ref features) = batch.features {
                let total_cells = features.len() * features.get(0).map_or(0, |row| row.len());
                let non_zero_cells = features
                    .iter()
                    .flat_map(|row| row.iter())
                    .filter(|&val| *val != 0.0)
                    .count();
                
                if total_cells > 0 {
                    self.quality_metrics.completeness_score = (non_zero_cells as f64 / total_cells as f64) * 100.0;
                }
                
                // 计算一致性评分(基于数据的方差)
                let all_values: Vec<f32> = features.iter().flatten().copied().collect();
                if !all_values.is_empty() {
                    let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
                    let variance = all_values
                        .iter()
                        .map(|x| {
                            let diff = *x - mean;
                            diff * diff
                        })
                        .sum::<f32>()
                        / all_values.len() as f32;
                    let std_dev = variance.sqrt();
                    
                    // 标准差越小，一致性越高
                    self.quality_metrics.consistency_score = ((1.0 / (1.0 + std_dev)) * 100.0) as f64;
                }
            }
            
            // 设置默认评分
            self.quality_metrics.accuracy_score = 85.0;
            self.quality_metrics.timeliness_score = 90.0;
            
            // 计算总体评分
            self.quality_metrics.calculate_overall_score();
        }
    }
    
    /// 高级处理配置
    #[derive(Debug, Clone)]
    pub struct AdvancedProcessingConfig {
        /// 是否启用质量检查
        pub enable_quality_check: bool,
        /// 是否启用异常检测
        pub enable_anomaly_detection: bool,
        /// 异常检测阈值
        pub anomaly_threshold: f64,
        /// 是否启用自动修复
        pub enable_auto_repair: bool,
        /// 自定义处理规则
        pub custom_rules: Vec<ProcessingRule>,
    }
    
    impl Default for AdvancedProcessingConfig {
        fn default() -> Self {
            Self {
                enable_quality_check: true,
                enable_anomaly_detection: false,
                anomaly_threshold: 3.0,
                enable_auto_repair: false,
                custom_rules: Vec::new(),
            }
        }
    }
    
    /// 处理规则
    #[derive(Debug, Clone)]
    pub struct ProcessingRule {
        /// 规则名称
        pub name: String,
        /// 规则条件
        pub condition: String,
        /// 规则动作
        pub action: String,
        /// 规则参数
        pub parameters: HashMap<String, String>,
    }
}

/// 导出配置 - 与handlers.rs兼容
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportConfig {
    /// 导出格式
    pub format: String,
    /// 导出路径
    pub output_path: String,
    /// 是否压缩
    pub compress: bool,
    /// 导出选项
    pub options: HashMap<String, String>,
    /// 字段选择
    pub fields: Option<Vec<String>>,
    /// 过滤条件
    pub filter: Option<String>,
    /// 最大导出记录数
    pub max_records: Option<usize>,
}

impl ExportConfig {
    /// 创建新的导出配置
    pub fn new(format: String, output_path: String) -> Self {
        Self {
            format,
            output_path,
            compress: false,
            options: HashMap::new(),
            fields: None,
            filter: None,
            max_records: None,
        }
    }

    /// 设置压缩选项
    pub fn with_compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    /// 添加选项
    pub fn with_option(mut self, key: String, value: String) -> Self {
        self.options.insert(key, value);
        self
    }

    /// 设置字段选择
    pub fn with_fields(mut self, fields: Vec<String>) -> Self {
        self.fields = Some(fields);
        self
    }

    /// 设置过滤条件
    pub fn with_filter(mut self, filter: String) -> Self {
        self.filter = Some(filter);
        self
    }

    /// 设置最大记录数
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = Some(max_records);
        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        if self.format.is_empty() {
            return Err(Error::config("导出格式不能为空".to_string()));
        }

        if self.output_path.is_empty() {
            return Err(Error::config("导出路径不能为空".to_string()));
        }

        // 验证格式支持
        match self.format.to_lowercase().as_str() {
            "csv" | "json" | "parquet" | "excel" | "xml" => {},
            _ => return Err(Error::config(format!("不支持的导出格式: {}", self.format))),
        }

        Ok(())
    }

    /// 获取输出文件扩展名
    pub fn get_extension(&self) -> &str {
        match self.format.to_lowercase().as_str() {
            "csv" => ".csv",
            "json" => ".json", 
            "parquet" => ".parquet",
            "excel" => ".xlsx",
            "xml" => ".xml",
            _ => ".txt",
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self::new("csv".to_string(), "output.csv".to_string())
    }
}
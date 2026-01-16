// 数据批次模块
// 处理数据分批和批量操作

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use std::collections::VecDeque;

use crate::error::{Result, Error};
use crate::data::schema::DataSchema;
use crate::data::value::DataValue;
use crate::data::types::{DataFormat, DataStatus};
use crate::data::loader::DataLoader;
use crate::data::processor::DataType;

// Stub type for validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid,
    Warning,
}

/// 数据批次状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStatus {
    /// 初始化
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
    /// 已销毁
    Disposed,
}

impl fmt::Display for BatchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchStatus::Initialized => write!(f, "初始化"),
            BatchStatus::Loading => write!(f, "加载中"),
            BatchStatus::Loaded => write!(f, "已加载"),
            BatchStatus::Processing => write!(f, "处理中"),
            BatchStatus::Processed => write!(f, "已处理"),
            BatchStatus::Error(msg) => write!(f, "错误: {}", msg),
            BatchStatus::Disposed => write!(f, "已销毁"),
        }
    }
}

/// 批次元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetadata {
    /// 批次创建时间
    pub created_at: DateTime<Utc>,
    /// 批次更新时间
    pub updated_at: DateTime<Utc>,
    /// 批次来源
    pub source: Option<String>,
    /// 批次处理信息
    pub processing_info: HashMap<String, String>,
    /// 自定义属性
    pub properties: HashMap<String, String>,
}

impl Default for BatchMetadata {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            updated_at: Utc::now(),
            source: None,
            processing_info: HashMap::new(),
            properties: HashMap::new(),
        }
    }
}

/// 数据批次
/// 
/// 数据批次是一组数据记录的集合，用于批量处理数据。
/// 每个批次包含一组记录和相关的元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBatch {
    /// 批次ID
    pub id: Option<String>,
    /// 数据集ID
    pub dataset_id: String,
    /// 批次索引
    pub index: usize,
    /// 批次索引（别名字段，为了兼容性）
    pub batch_index: usize,
    /// 批次大小
    pub size: usize,
    /// 批次大小（别名字段，为了兼容性）
    pub batch_size: usize,
    /// 批次状态
    pub status: DataStatus,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 数据内容 (可选)
    pub data: Option<Vec<u8>>,
    /// 标签数据 (可选)
    pub labels: Option<Vec<u8>>,
    /// 元数据信息
    pub metadata: HashMap<String, String>,
    /// 数据格式
    pub format: DataFormat,
    /// 数据源
    pub source: Option<String>,
    /// 数据记录
    pub records: Vec<HashMap<String, DataValue>>,
    /// 数据模式
    pub schema: Option<DataSchema>,
    /// 字段名列表
    pub field_names: Vec<String>,
    /// 特征数据
    pub features: Option<Vec<Vec<f32>>>,
    /// 目标数据
    pub target: Option<Vec<f32>>,
    /// 数据版本
    pub version: Option<u32>,
    /// 数据校验和
    pub checksum: Option<String>,
    /// 压缩信息
    pub compression: Option<String>,
    /// 加密信息
    pub encryption: Option<String>,
    /// 标签列表
    pub tags: Vec<String>,
    /// 验证状态
    pub validation_status: Option<ValidationStatus>,
    /// 验证错误列表
    pub validation_errors: Vec<String>,
    /// 处理时间
    pub processing_time: std::time::Duration,
    /// 质量分数
    pub quality_score: Option<f32>,
    /// 父批次ID
    pub parent_batch_id: Option<String>,
    /// 子批次ID列表
    pub child_batch_ids: Vec<String>,
    /// 依赖关系
    pub dependencies: Vec<String>,
    /// 优先级
    pub priority: i32,
    /// 重试次数
    pub retry_count: u32,
    /// 最大重试次数
    pub max_retries: u32,
    /// 超时设置
    pub timeout: Option<std::time::Duration>,
    /// 错误消息
    pub error_message: Option<String>,
    /// 处理结果
    pub result: Option<String>,
    /// 自定义数据
    pub custom_data: HashMap<String, serde_json::Value>,
}

impl Default for DataBatch {
    fn default() -> Self {
        Self {
            id: Some(Uuid::new_v4().to_string()),
            dataset_id: "default".to_string(),
            index: 0,
            batch_index: 0,
            size: 0,
            batch_size: 0,
            status: DataStatus::Created,
            created_at: Utc::now(),
            data: None,
            labels: None,
            metadata: HashMap::new(),
            format: DataFormat::Binary,
            source: None,
            records: Vec::new(),
            schema: None,
            field_names: Vec::new(),
            features: None,
            target: None,
            version: None,
            checksum: None,
            compression: None,
            encryption: None,
            tags: Vec::new(),
            validation_status: None,
            validation_errors: Vec::new(),
            processing_time: std::time::Duration::from_secs(0),
            quality_score: None,
            parent_batch_id: None,
            child_batch_ids: Vec::new(),
            dependencies: Vec::new(),
            priority: 0,
            retry_count: 0,
            max_retries: 3,
            timeout: None,
            error_message: None,
            result: None,
            custom_data: HashMap::new(),
        }
    }
}

impl DataBatch {
    /// 创建新的数据批次
    pub fn new(dataset_id: &str, index: usize, size: usize) -> Self {
        Self {
            id: Some(Uuid::new_v4().to_string()),
            dataset_id: dataset_id.to_string(),
            index,
            batch_index: index,  // 添加别名字段
            size,
            batch_size: size,    // 添加别名字段
            status: DataStatus::Created,
            created_at: Utc::now(),
            data: None,
            labels: None,
            metadata: HashMap::new(),
            format: DataFormat::Binary,
            source: None,
            records: Vec::new(),
            schema: None,
            field_names: Vec::new(),
            features: None,
            target: None,
            version: None,
            checksum: None,
            compression: None,
            encryption: None,
            tags: Vec::new(),
            validation_status: None,
            validation_errors: Vec::new(),
            processing_time: std::time::Duration::from_secs(0),
            quality_score: None,
            parent_batch_id: None,
            child_batch_ids: Vec::new(),
            dependencies: Vec::new(),
            priority: 0,
            retry_count: 0,
            max_retries: 3,
            timeout: None,
            error_message: None,
            result: None,
            custom_data: HashMap::new(),
        }
    }

    /// 获取批次ID（API兼容性）
    pub fn batch_id(&self) -> Option<&String> {
        self.id.as_ref()
    }

    /// 获取样本数量
    pub fn num_samples(&self) -> usize {
        if !self.records.is_empty() {
            self.records.len()
        } else if let Some(ref data) = self.data {
            // 对于二进制数据，假设每个样本为32字节（可以根据实际情况调整）
            data.len() / 32
        } else {
            self.size
        }
    }

    /// 更新状态
    pub fn update_status(&mut self, status: DataStatus) {
        self.status = status;
    }

    /// 设置数据
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = Some(data);
        self
    }

    /// 设置标签
    pub fn with_labels(mut self, labels: Vec<u8>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// 设置格式
    pub fn with_format(mut self, format: DataFormat) -> Self {
        self.format = format;
        self
    }

    /// 设置数据源
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// 设置特征数据（同时更新批次大小）
    pub fn with_features(mut self, features: Vec<Vec<f32>>) -> Self {
        let len = features.len();
        self.features = Some(features);
        self.size = len;
        self.batch_size = len;
        self
    }

    /// 数据标准化
    pub fn normalize(mut self) -> Self {
        // 对数值字段进行标准化
        for record in &mut self.records {
            for (_, value) in record.iter_mut() {
                if let DataValue::Float(f) = value {
                    // 简单的标准化：将值缩放到[0,1]范围
                    *f = (*f - 0.0) / 1.0; // 这里应该根据实际数据范围进行标准化
                }
            }
        }
        self
    }

    /// 数据打乱
    pub fn shuffle(mut self) -> Self {
        use fastrand::Rng;
        let mut rng = Rng::new();
        
        // 打乱记录顺序
        for i in (1..self.records.len()).rev() {
            let j = rng.usize(..=i);
            self.records.swap(i, j);
        }
        self
    }

    /// 分割验证集
    pub fn split_validation(mut self, validation_split: f32) -> Vec<Self> {
        let total_records = self.records.len();
        let validation_size = (total_records as f32 * validation_split) as usize;
        let training_size = total_records - validation_size;
        
        // 分割记录
        let validation_records = self.records.split_off(training_size);
        
        // 创建训练批次
        let mut training_batch = self.clone();
        training_batch.records = self.records;
        training_batch.size = training_size;
        training_batch.batch_size = training_size;
        
        // 创建验证批次
        let mut validation_batch = DataBatch::new(&self.dataset_id, 1, validation_size);
        validation_batch.id = Some(uuid::Uuid::new_v4().to_string());
        validation_batch.schema = self.schema.clone();
        validation_batch.data = Some(Vec::new());
        validation_batch.labels = self.labels.clone();
        validation_batch.metadata = self.metadata.clone();
        validation_batch.created_at = self.created_at;
        validation_batch.records = validation_records;
        validation_batch.size = validation_size;
        validation_batch.batch_size = validation_size;
        validation_batch.id = Some(uuid::Uuid::new_v4().to_string());
        
        vec![training_batch, validation_batch]
    }

    /// 添加JSON记录
    pub fn add_record_json(&mut self, json_value: serde_json::Value) -> Result<()> {
        let mut record = HashMap::new();
        
        if let serde_json::Value::Object(obj) = json_value {
            for (key, value) in obj {
                let data_value = match value {
                    serde_json::Value::Null => DataValue::Null,
                    serde_json::Value::Bool(b) => DataValue::Boolean(b),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            DataValue::Integer(i)
                        } else if let Some(f) = n.as_f64() {
                            DataValue::Float(f)
                        } else {
                            DataValue::Null
                        }
                    },
                    serde_json::Value::String(s) => DataValue::Text(s),
                    serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                        DataValue::Text(value.to_string())
                    }
                };
                record.insert(key, data_value);
            }
        }
        
        self.records.push(record);
        self.size = self.records.len();
        self.batch_size = self.size;
        
        Ok(())
    }

    /// 获取特征数据
    pub fn get_features(&self) -> Result<Vec<Vec<f32>>> {
        if self.records.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut features = Vec::new();
        for record in &self.records {
            let mut feature_vec = Vec::new();
            
            // 遍历记录中的所有字段（除了标签字段）
            for (key, value) in record {
                if !key.starts_with("label") && !key.starts_with("target") {
                    match value {
                        DataValue::Float(f) => feature_vec.push(*f as f32),
                        DataValue::Integer(i) => feature_vec.push(*i as f32),
                        DataValue::Boolean(b) => feature_vec.push(if *b { 1.0 } else { 0.0 }),
                        _ => feature_vec.push(0.0), // 其他类型默认为0
                    }
                }
            }
            
            if !feature_vec.is_empty() {
                features.push(feature_vec);
            }
        }
        
        Ok(features)
    }
    
    /// 获取特征数据（便捷方法，返回Vec<Vec<f32>>）
    pub fn features(&self) -> Vec<Vec<f32>> {
        // 优先使用已存储的features字段
        if let Some(ref stored_features) = self.features {
            return stored_features.clone();
        }
        
        // 否则从records中提取
        self.get_features().unwrap_or_default()
    }
    
    /// 获取元数据（便捷方法，返回&HashMap<String, String>）
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    /// 获取标签数据
    pub fn get_labels(&self) -> Result<Vec<Vec<f32>>> {
        if self.records.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut labels = Vec::new();
        for record in &self.records {
            let mut label_vec = Vec::new();
            
            // 查找标签字段
            for (key, value) in record {
                if key.starts_with("label") || key.starts_with("target") {
                    match value {
                        DataValue::Float(f) => label_vec.push(*f as f32),
                        DataValue::Integer(i) => label_vec.push(*i as f32),
                        DataValue::Boolean(b) => label_vec.push(if *b { 1.0 } else { 0.0 }),
                        _ => label_vec.push(0.0), // 其他类型默认为0
                    }
                }
            }
            
            // 如果没有找到标签字段，创建默认标签
            if label_vec.is_empty() {
                label_vec.push(0.0);
            }
            
            labels.push(label_vec);
        }
        
        Ok(labels)
    }

    /// 获取特征数据（与get_features相同，为了兼容性）
    pub fn get_feature_data(&self) -> Result<Vec<Vec<f32>>> {
        self.get_features()
    }

    /// 获取标签数据（与get_labels相同，为了兼容性）
    pub fn get_label_data(&self) -> Result<Vec<Vec<f32>>> {
        self.get_labels()
    }

    /// 检查批次是否为空
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// 获取批次中记录的数量
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// 获取记录的可变引用（用于兼容性）
    pub fn records(&self) -> &Vec<HashMap<String, DataValue>> {
        &self.records
    }

    /// 获取记录的可变引用（用于兼容性）
    pub fn records_mut(&mut self) -> &mut Vec<HashMap<String, DataValue>> {
        &mut self.records
    }

    /// 获取批次的schema
    pub fn schema(&self) -> Option<&DataSchema> {
        self.schema.as_ref()
    }

    /// 设置批次的schema
    pub fn set_schema(&mut self, schema: DataSchema) {
        self.schema = Some(schema);
    }

    /// 获取字段名列表
    pub fn get_field_names(&self) -> &Vec<String> {
        &self.field_names
    }

    /// 设置字段名列表
    pub fn set_field_names(&mut self, field_names: Vec<String>) {
        self.field_names = field_names;
    }

    /// 添加字段名
    pub fn add_field_name(&mut self, field_name: String) {
        if !self.field_names.contains(&field_name) {
            self.field_names.push(field_name);
        }
    }

    /// 从记录中自动提取字段名
    pub fn extract_field_names_from_records(&mut self) {
        let mut field_names = std::collections::HashSet::new();
        for record in &self.records {
            for key in record.keys() {
                field_names.insert(key.clone());
            }
        }
        self.field_names = field_names.into_iter().collect();
        self.field_names.sort(); // 保持字段名的一致顺序
    }

    /// 使用字段名创建批次
    pub fn with_field_names(mut self, field_names: Vec<String>) -> Self {
        self.field_names = field_names;
        self
    }

    /// 批次大小
    pub fn batch_size(&self) -> usize {
        self.records.len()
    }

    /// 切片批次数据
    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if start >= self.records.len() || end > self.records.len() || start >= end {
            return Err(Error::invalid_input("Invalid slice range"));
        }

        let mut sliced_batch = self.clone();
        sliced_batch.records = self.records[start..end].to_vec();
        sliced_batch.size = sliced_batch.records.len();
        Ok(sliced_batch)
    }

    /// 从记录创建批次
    pub fn from_records(
        records: Vec<HashMap<String, crate::data::DataValue>>,
        schema: Option<DataSchema>
    ) -> Result<Self> {
        let mut batch = Self::new("imported", 0, records.len());
        batch.records = records;
        batch.schema = schema;
        batch.status = DataStatus::Loaded;
        
        // 自动提取字段名
        batch.extract_field_names_from_records();
        
        // 添加元数据
        batch.metadata.insert("source".to_string(), "imported".to_string());
        batch.metadata.insert("record_count".to_string(), batch.records.len().to_string());
        batch.metadata.insert("field_count".to_string(), batch.field_names.len().to_string());
        
        Ok(batch)
    }

    /// 转换为数据记录
    pub fn to_records(&self) -> Vec<crate::data::record::DataRecord> {
        self.records.iter().map(|record_map| {
            let mut record = crate::data::record::DataRecord::new();
            for (field_name, field_value) in record_map {
                let value = crate::data::record::Value::Data(field_value.clone());
                record.add_field(field_name, value);
            }
            record
        }).collect()
    }

    /// 从DataRecord创建批次
    pub fn from_data_records(
        records: Vec<crate::data::record::DataRecord>,
        schema: Option<DataSchema>
    ) -> Result<Self> {
        let converted_records: Vec<HashMap<String, crate::data::DataValue>> = records
            .into_iter()
            .map(|record| {
                let mut record_map = HashMap::new();
                for (field_name, field_value) in record.fields {
                    if let Ok(data_value) = field_value.to_data_value() {
                        record_map.insert(field_name, data_value);
                    }
                }
                record_map
            })
            .collect();
        
        Self::from_records(converted_records, schema)
    }

    /// 从字节数据创建批次
    pub fn from_bytes(data: &[u8], data_type: DataType) -> Result<Self> {
        let mut batch = Self::new("default", 0, data.len());
        batch.id = Some(Uuid::new_v4().to_string());
        batch.status = DataStatus::Loaded;
        batch.data = Some(data.to_vec());
        batch.metadata.insert("data_type".to_string(), format!("{:?}", data_type));
        batch.metadata.insert("encoding".to_string(), "binary".to_string());
        batch.format = DataFormat::Binary;
        batch.source = Some("bytes".to_string());
        
        Ok(batch)
    }

    /// 从二进制数据创建批次
    pub fn from_binary(data: Vec<u8>, data_type: DataType) -> Self {
        let mut batch = Self::new("binary", 0, data.len());
        batch.id = Some(Uuid::new_v4().to_string());
        batch.status = DataStatus::Loaded;
        batch.data = Some(data);
        batch.metadata.insert("data_type".to_string(), format!("{:?}", data_type));
        batch.format = DataFormat::Binary;
        batch.source = Some("binary".to_string());
        batch
    }

    /// 从原始数据创建数据批次
    pub fn from_raw_data(raw_data: &[u8]) -> Result<Self> {
        // 尝试解析为JSON格式
        if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(raw_data) {
            return Self::from_json_value(json_value);
        }
        
        // 尝试解析为CSV格式
        if let Ok(csv_string) = std::str::from_utf8(raw_data) {
            return Self::from_csv_string(csv_string);
        }
        
        // 如果都不是，则作为二进制数据处理
        Ok(Self::from_binary(raw_data.to_vec(), DataType::Binary))
    }

    /// 从JSON值创建数据批次
    fn from_json_value(json_value: serde_json::Value) -> Result<Self> {
        let mut batch = Self::default();
        
        match json_value {
            serde_json::Value::Array(records) => {
                for record in records {
                    if let serde_json::Value::Object(obj) = record {
                        let mut data_record = HashMap::new();
                        for (key, value) in obj {
                            data_record.insert(key, Self::json_value_to_data_value(value));
                        }
                        batch.records.push(data_record);
                    }
                }
            },
            serde_json::Value::Object(obj) => {
                // 单个对象，转换为记录
                let mut data_record = HashMap::new();
                for (key, value) in obj {
                    data_record.insert(key, Self::json_value_to_data_value(value));
                }
                batch.records.push(data_record);
            },
            _ => {
                return Err(Error::invalid_input("不支持的JSON格式"));
            }
        }
        
        batch.size = batch.records.len();
        batch.batch_size = batch.records.len();
        batch.format = DataFormat::Json;
        
        // 提取字段名
        if !batch.records.is_empty() {
            batch.field_names = batch.records[0].keys().cloned().collect();
        }
        
        Ok(batch)
    }

    /// 从CSV字符串创建数据批次
    fn from_csv_string(csv_string: &str) -> Result<Self> {
        let mut batch = Self::default();
        let mut reader = csv::Reader::from_reader(csv_string.as_bytes());
        
        // 读取字段名
        if let Ok(headers) = reader.headers() {
            batch.field_names = headers.iter().map(|s| s.to_string()).collect();
        }
        
        // 读取记录
        for result in reader.records() {
            match result {
                Ok(record) => {
                    let mut data_record = HashMap::new();
                    for (i, field) in record.iter().enumerate() {
                        let field_name = if i < batch.field_names.len() {
                            batch.field_names[i].clone()
                        } else {
                            format!("field_{}", i)
                        };
                        data_record.insert(field_name, Self::parse_csv_value(field));
                    }
                    batch.records.push(data_record);
                },
                Err(e) => {
                    return Err(Error::invalid_input(format!("CSV解析错误: {}", e)));
                }
            }
        }
        
        batch.size = batch.records.len();
        batch.batch_size = batch.records.len();
        batch.format = DataFormat::Csv;
        
        Ok(batch)
    }

    /// 将JSON值转换为DataValue
    fn json_value_to_data_value(value: serde_json::Value) -> crate::data::DataValue {
        match value {
            serde_json::Value::Null => crate::data::DataValue::Null,
            serde_json::Value::Bool(b) => crate::data::DataValue::Boolean(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    crate::data::DataValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    crate::data::DataValue::Float(f)
                } else {
                    crate::data::DataValue::String(n.to_string())
                }
            },
            serde_json::Value::String(s) => crate::data::DataValue::String(s),
            serde_json::Value::Array(arr) => {
                let values: Vec<crate::data::DataValue> = arr.into_iter()
                    .map(Self::json_value_to_data_value)
                    .collect();
                crate::data::DataValue::Array(values)
            },
            serde_json::Value::Object(obj) => {
                let values: HashMap<String, crate::data::DataValue> = obj.into_iter()
                    .map(|(k, v)| (k, Self::json_value_to_data_value(v)))
                    .collect();
                crate::data::DataValue::Object(values)
            },
        }
    }

    /// 解析CSV值
    fn parse_csv_value(value: &str) -> crate::data::DataValue {
        // 尝试解析为数字
        if let Ok(i) = value.parse::<i64>() {
            return crate::data::DataValue::Integer(i);
        }
        if let Ok(f) = value.parse::<f64>() {
            return crate::data::DataValue::Float(f);
        }
        
        // 尝试解析为布尔值
        match value.to_lowercase().as_str() {
            "true" | "yes" | "1" => return crate::data::DataValue::Boolean(true),
            "false" | "no" | "0" => return crate::data::DataValue::Boolean(false),
            _ => {}
        }
        
        // 默认为字符串
        crate::data::DataValue::String(value.to_string())
    }

    /// 创建带有模式的空数据批次
    pub fn empty_with_schema(schema: Option<DataSchema>) -> Self {
        let mut batch = Self::new("empty", 0, 0);
        batch.schema = schema;
        batch
    }

    /// 转换为字节数据
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.data.clone().ok_or_else(|| {
            crate::error::Error::invalid_data("No data in batch".to_string())
        })
    }

    /// 将数据批次转换为张量数据并分离特征和标签
    pub fn to_tensor_data_split(&self) -> Result<(Vec<crate::compat::tensor::TensorData>, Vec<crate::compat::tensor::TensorData>)> {
        use crate::compat::tensor::TensorData;
        
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        // 如果存在二进制数据和标签数据，优先使用
        if let (Some(data_bytes), Some(label_bytes)) = (&self.data, &self.labels) {
            // 尝试反序列化二进制数据
            if let Ok(feature_tensor) = bincode::deserialize::<TensorData>(data_bytes) {
                features.push(feature_tensor);
            }
            if let Ok(label_tensor) = bincode::deserialize::<TensorData>(label_bytes) {
                labels.push(label_tensor);
            }
            
            if !features.is_empty() && !labels.is_empty() {
                return Ok((features, labels));
            }
        }
        
        // 如果没有二进制数据，从记录中提取
        if !self.records.is_empty() {
            let num_records = self.records.len();
            
            // 推断特征和标签列
            let mut feature_columns = Vec::new();
            let mut label_columns = Vec::new();
            
            // 简单的特征标签分离策略：
            // 1. 如果有'label'、'target'、'y'等列名，则作为标签
            // 2. 其他数值列作为特征
            if let Some(first_record) = self.records.first() {
                for (column_name, _) in first_record {
                    if column_name.to_lowercase().contains("label") 
                        || column_name.to_lowercase().contains("target")
                        || column_name.to_lowercase() == "y"
                        || column_name.to_lowercase().contains("class") {
                        label_columns.push(column_name.clone());
                    } else {
                        feature_columns.push(column_name.clone());
                    }
                }
            }
            
            // 如果没有明确的标签列，假设最后一列为标签
            if label_columns.is_empty() && !feature_columns.is_empty() {
                if let Some(last_column) = feature_columns.pop() {
                    label_columns.push(last_column);
                }
            }
            
            // 提取特征数据
            if !feature_columns.is_empty() {
                let mut feature_data = Vec::new();
                for record in &self.records {
                    let mut row_features = Vec::new();
                    for column in &feature_columns {
                        if let Some(value) = record.get(column) {
                            let numeric_value = match value {
                                DataValue::Integer(i) => *i as f32,
                                DataValue::Float(f) => *f as f32,
                                DataValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                                DataValue::Text(s) => {
                                    // 尝试解析文本为数字
                                    s.parse::<f32>().unwrap_or(0.0)
                                },
                                _ => 0.0,
                            };
                            row_features.push(numeric_value);
                        }
                    }
                    feature_data.extend(row_features);
                }
                
                let feature_tensor = crate::compat::tensor::TensorData {
                    shape: vec![num_records, feature_columns.len()],
                    data: crate::compat::tensor::TensorValues::F32(feature_data),
                    dtype: crate::compat::tensor::DataType::Float32,
                    metadata: HashMap::new(),
                };
                features.push(feature_tensor);
            }
            
            // 提取标签数据
            if !label_columns.is_empty() {
                let mut label_data = Vec::new();
                for record in &self.records {
                    let mut row_labels = Vec::new();
                    for column in &label_columns {
                        if let Some(value) = record.get(column) {
                            let numeric_value = match value {
                                DataValue::Integer(i) => *i as f32,
                                DataValue::Float(f) => *f as f32,
                                DataValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                                DataValue::Text(s) => {
                                    // 尝试解析文本为数字，或者进行标签编码
                                    s.parse::<f32>().unwrap_or_else(|_| {
                                        // 简单的字符串哈希编码
                                        (s.chars().map(|c| c as u32).sum::<u32>() % 1000) as f32
                                    })
                                },
                                _ => 0.0,
                            };
                            row_labels.push(numeric_value);
                        }
                    }
                    label_data.extend(row_labels);
                }
                
                let label_tensor = crate::compat::tensor::TensorData {
                    shape: vec![num_records, label_columns.len()],
                    data: crate::compat::tensor::TensorValues::F32(label_data),
                    dtype: crate::compat::tensor::DataType::Float32,
                    metadata: HashMap::new(),
                };
                labels.push(label_tensor);
            }
        }
        
        // 如果仍然没有数据，创建默认的空张量
        if features.is_empty() {
            features.push(crate::compat::tensor::TensorData {
                shape: vec![0, 0],
                data: crate::compat::tensor::TensorValues::F32(Vec::new()),
                dtype: crate::compat::tensor::DataType::Float32,
                metadata: HashMap::new(),
            });
        }
        
        if labels.is_empty() {
            labels.push(crate::compat::tensor::TensorData {
                shape: vec![0, 0],
                data: crate::compat::tensor::TensorValues::F32(Vec::new()),
                dtype: crate::compat::tensor::DataType::Float32,
                metadata: HashMap::new(),
            });
        }
        
        Ok((features, labels))
    }

    /// 从ManagerDataset创建DataBatch
    pub fn from_manager_dataset(dataset: crate::data::manager::ManagerDataset) -> Result<Self> {
        let mut batch = Self::new(&dataset.id, 0, dataset.size);
        
        // 设置元数据
        batch.metadata = dataset.metadata.clone();
        batch.add_metadata("name", &dataset.name);
        batch.add_metadata("format", &dataset.format);
        batch.add_metadata("dataset_type", &format!("{:?}", dataset.dataset_type));
        batch.add_metadata("created_at", &dataset.created_at.to_string());
        batch.add_metadata("updated_at", &dataset.updated_at.to_string());
        
        // 设置数据格式
        match dataset.format.as_str() {
            "json" => batch.format = DataFormat::Json,
            "csv" => batch.format = DataFormat::Csv,
            "binary" | "bin" => batch.format = DataFormat::Binary,
            "text" | "txt" => batch.format = DataFormat::Text,
            _ => batch.format = DataFormat::Binary,
        }
        
        // 设置状态
        batch.status = DataStatus::Loaded;
        
        Ok(batch)
    }
    
    /// 合并另一个批次到当前批次
    pub fn merge(&mut self, other: &DataBatch) -> Result<()> {
        // 合并记录
        self.records.extend(other.records.clone());
        
        // 更新大小
        self.size = self.records.len();
        self.batch_size = self.size;
        
        // 合并元数据
        for (key, value) in &other.metadata {
            self.metadata.insert(key.clone(), value.clone());
        }
        
        // 合并字段名
        for field_name in &other.field_names {
            if !self.field_names.contains(field_name) {
                self.field_names.push(field_name.clone());
            }
        }
        
        // 合并标签
        self.tags.extend(other.tags.clone());
        
        // 合并验证错误
        self.validation_errors.extend(other.validation_errors.clone());
        
        // 合并子批次ID
        self.child_batch_ids.extend(other.child_batch_ids.clone());
        
        // 合并依赖关系
        self.dependencies.extend(other.dependencies.clone());
        
        // 合并自定义数据
        for (key, value) in &other.custom_data {
            self.custom_data.insert(key.clone(), value.clone());
        }
        
        // 如果另一个批次有特征数据，尝试合并
        if let Some(ref other_features) = other.features {
            if let Some(ref mut self_features) = self.features {
                self_features.extend(other_features.clone());
            } else {
                self.features = Some(other_features.clone());
            }
        }
        
        Ok(())
    }
}

/// 数据迭代器
#[derive(Clone)]
pub struct DataIterator {
    loader: Arc<dyn DataLoader>,
    path: String,
    batch_size: usize,
    current_index: usize,
    total_size: usize,
}

impl DataIterator {
    /// 创建新的数据迭代器
    pub fn new(loader: Arc<dyn DataLoader>, path: String, batch_size: usize) -> Self {
        Self {
            loader,
            path,
            batch_size,
            current_index: 0,
            total_size: 0, // 需要在初始化时计算
        }
    }

    /// 重置迭代器
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// 获取总大小
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// 获取批次大小
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 获取当前索引
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// 获取剩余批次数
    pub fn remaining_batches(&self) -> usize {
        if self.total_size == 0 {
            return 0;
        }
        let remaining_items = self.total_size.saturating_sub(self.current_index);
        (remaining_items + self.batch_size - 1) / self.batch_size
    }

    /// 获取下一个批次
    pub async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        if self.current_index >= self.total_size {
            return Ok(None);
        }

        // 计算批次大小
        let batch_size = std::cmp::min(
            self.batch_size,
            self.total_size - self.current_index
        );

        // 加载数据
        let source = crate::data::loader::DataSource::File(self.path.clone());
        let format = crate::data::loader::types::DataFormat::json();
        match self.loader.load_batch(&source, &format, batch_size, self.current_index).await {
            Ok(batch) => {
                self.current_index += batch_size;
                Ok(Some(batch))
            }
            Err(e) => Err(e),
        }
    }
}

/// 数据批次扩展trait
pub trait DataBatchExt {
    fn get_id(&self) -> String;
    fn get_dataset_id(&self) -> Option<String>;
    fn get_index(&self) -> Option<usize>;
    fn get_size(&self) -> usize;
    fn get_status(&self) -> Option<DataStatus>;
    fn get_creation_time(&self) -> Option<DateTime<Utc>>;
}

impl DataBatchExt for DataBatch {
    fn get_id(&self) -> String {
        self.id.clone().unwrap_or_else(|| "unknown".to_string())
    }

    fn get_dataset_id(&self) -> Option<String> {
        Some(self.dataset_id.clone())
    }

    fn get_index(&self) -> Option<usize> {
        Some(self.index)
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_status(&self) -> Option<DataStatus> {
        Some(self.status.clone())
    }

    fn get_creation_time(&self) -> Option<DateTime<Utc>> {
        Some(self.created_at)
    }
}

/// 数据源配置
#[derive(Clone)]
pub struct DataSourceConfig {
    pub name: String,
    pub path: std::path::PathBuf,
    pub format: DataFormat,
    pub loader: Arc<dyn DataLoader>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub cache: bool,
}

/// 基础数据源
#[derive(Clone)]
pub struct BasicDataSource {
    pub name: String,
    pub path: std::path::PathBuf,
    pub format: DataFormat,
    pub schema: Option<crate::data::schema::DataSchema>,
    pub loader: Arc<dyn DataLoader>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub cache: bool,
    pub total_size: usize,
    pub loaded_size: usize,
    pub current_index: usize,
    pub batches: VecDeque<DataBatch>,
}

impl BasicDataSource {
    /// 创建新的基础数据源
    pub fn new(loader: Arc<dyn DataLoader>, path: String, batch_size: usize) -> Self {
        Self {
            name: "default".to_string(),
            path: std::path::PathBuf::from(path),
            format: DataFormat::CSV,
            schema: None,
            loader,
            batch_size,
            shuffle: false,
            cache: false,
            total_size: 0,
            loaded_size: 0,
            current_index: 0,
            batches: VecDeque::new(),
        }
    }
}


/// 批次处理器特征
pub trait BatchProcessor {
    /// 处理数据批次
    fn process(&self, batch: &mut DataBatch) -> Result<()>;
    
    /// 处理器名称
    fn name(&self) -> &str;
    
    /// 处理器描述
    fn description(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_batch_creation() {
        let batch = DataBatch::new("default", 0, 0);
        assert_eq!(batch.status, DataStatus::Created);
        assert_eq!(batch.size, 0);
        assert_eq!(batch.index, 0);
        assert_eq!(batch.dataset_id, "default");
        assert!(batch.data.is_none());
        assert!(batch.labels.is_none());
        assert!(batch.metadata.is_empty());
        assert_eq!(batch.format, DataFormat::Binary);
        assert!(batch.source.is_none());
    }
    
    #[test]
    fn test_update_status() {
        let mut batch = DataBatch::new("default", 0, 0);
        batch.update_status(DataStatus::Loaded);
        assert_eq!(batch.status, DataStatus::Loaded);
    }
    
    #[test]
    fn test_set_data() {
        let mut batch = DataBatch::new("default", 0, 0);
        let data = vec![1, 2, 3];
        batch.with_data(data);
        assert_eq!(batch.data, Some(vec![1, 2, 3]));
    }
    
    #[test]
    fn test_set_labels() {
        let mut batch = DataBatch::new("default", 0, 0);
        let labels = vec![1, 2, 3];
        batch.with_labels(labels);
        assert_eq!(batch.labels, Some(vec![1, 2, 3]));
    }
    
    #[test]
    fn test_set_metadata() {
        let mut batch = DataBatch::new("default", 0, 0);
        batch.with_metadata("key1", "value1");
        batch.with_metadata("key2", "value2");
        assert_eq!(batch.metadata.len(), 2);
        assert_eq!(batch.metadata["key1"], "value1");
        assert_eq!(batch.metadata["key2"], "value2");
    }
    
    #[test]
    fn test_set_format() {
        let mut batch = DataBatch::new("default", 0, 0);
        batch.with_format(DataFormat::CSV);
        assert_eq!(batch.format, DataFormat::CSV);
    }
    
    #[test]
    fn test_set_source() {
        let mut batch = DataBatch::new("default", 0, 0);
        batch.with_source("source1");
        assert_eq!(batch.source, Some("source1"));
    }
    
    #[test]
    fn test_from_bytes() {
        let data = vec![1, 2, 3];
        let batch = DataBatch::from_bytes(&data, DataType::Integer).unwrap();
        assert_eq!(batch.data, Some(vec![1, 2, 3]));
        assert_eq!(batch.format, DataFormat::Binary);
        assert_eq!(batch.source, Some("bytes"));
    }
    
    #[test]
    fn test_from_binary() {
        let data = vec![1, 2, 3];
        let batch = DataBatch::from_binary(data, DataType::Integer);
        assert_eq!(batch.data, Some(vec![1, 2, 3]));
        assert_eq!(batch.format, DataFormat::Binary);
        assert_eq!(batch.source, Some("binary"));
    }
    
    #[test]
    fn test_to_bytes() {
        let data = vec![1, 2, 3];
        let batch = DataBatch::from_bytes(&data, DataType::Integer).unwrap();
        let bytes = batch.to_bytes().unwrap();
        assert_eq!(bytes, vec![1, 2, 3]);
    }
} 
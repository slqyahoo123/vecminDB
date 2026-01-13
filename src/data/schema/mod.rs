// 数据schema模块 - 定义数据结构和模式

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::error::{Result, Error};

// 重新导出schema模块中的所有公共项
pub mod schema;
pub use schema::*;

// =============================================================================
// 数据库存储层类型系统 (Database Storage Types)
// =============================================================================

/// 数据库存储字段类型 - 用于底层数据存储
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageFieldType {
    /// 32位整数
    Int32,
    /// 64位整数  
    Int64,
    /// 32位浮点数
    Float32,
    /// 64位浮点数
    Float64,
    /// 字符串
    String,
    /// 布尔值
    Boolean,
    /// 日期
    Date,
    /// 时间
    Time,
    /// 日期时间
    DateTime,
    /// JSON对象
    Json,
    /// 二进制数据
    Binary,
    /// 数组类型
    Array(Box<StorageFieldType>),
    /// 自定义类型
    Custom(String),
}

/// AI/ML数据处理类型 - 用于AI模型训练和推理
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AIFieldType {
    /// 数值型数据 (可用于数学运算)
    Numeric,
    /// 文本数据 (用于NLP处理)
    Text,
    /// 分类数据 (离散标签)
    Categorical,
    /// 图像数据 (像素矩阵)
    Image,
    /// 音频数据 (音频信号)
    Audio,
    /// 视频数据 (视频帧序列)
    Video,
    /// 时间序列数据
    TimeSeries,
    /// 嵌入向量 (高维向量)
    Embedding(usize), // 维度
    /// 复合数据类型
    Composite(HashMap<String, AIFieldType>),
    /// 自定义AI类型
    Custom(String),
}

/// 统一字段类型 - 桥接存储类型和AI类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedFieldType {
    /// 存储类型
    Storage(StorageFieldType),
    /// AI处理类型
    AI(AIFieldType),
    /// 混合类型 (同时支持存储和AI处理)
    Hybrid {
        storage: StorageFieldType,
        ai: AIFieldType,
    },
}

// =============================================================================
// 向后兼容的FieldType定义
// =============================================================================

/// 向后兼容的FieldType - 映射到UnifiedFieldType
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    // 传统数据库类型
    Int,
    Integer,
    Float,
    String,
    Boolean,
    Date,
    Time,
    DateTime,
    Json,
    Object,
    Binary,
    Array(Box<FieldType>),
    
    // AI/ML类型
    Numeric,
    Text,
    Categorical,
    Image,
    Audio,
    Video,
    TimeSeries,
    Embedding(usize),
    
    // 自定义类型
    Custom(String),
}

impl FieldType {
    /// 转换为存储类型
    pub fn to_storage_type(&self) -> StorageFieldType {
        match self {
            FieldType::Int | FieldType::Integer => StorageFieldType::Int64,
            FieldType::Float => StorageFieldType::Float64,
            FieldType::String | FieldType::Text => StorageFieldType::String,
            FieldType::Boolean => StorageFieldType::Boolean,
            FieldType::Date => StorageFieldType::Date,
            FieldType::Time => StorageFieldType::Time,
            FieldType::DateTime => StorageFieldType::DateTime,
            FieldType::Json | FieldType::Object => StorageFieldType::Json,
            FieldType::Binary => StorageFieldType::Binary,
            FieldType::Array(inner) => StorageFieldType::Array(Box::new(inner.to_storage_type())),
            
            // AI类型映射到适当的存储类型
            FieldType::Numeric => StorageFieldType::Float64,
            FieldType::Categorical => StorageFieldType::String,
            FieldType::Image | FieldType::Audio | FieldType::Video => StorageFieldType::Binary,
            FieldType::TimeSeries => StorageFieldType::Json,
            FieldType::Embedding(_) => StorageFieldType::Binary,
            
            FieldType::Custom(name) => StorageFieldType::Custom(name.clone()),
        }
    }
    
    /// 转换为AI类型
    pub fn to_ai_type(&self) -> Option<AIFieldType> {
        match self {
            FieldType::Int | FieldType::Integer | FieldType::Float | FieldType::Numeric => {
                Some(AIFieldType::Numeric)
            },
            FieldType::String | FieldType::Text => Some(AIFieldType::Text),
            FieldType::Categorical => Some(AIFieldType::Categorical),
            FieldType::Image => Some(AIFieldType::Image),
            FieldType::Audio => Some(AIFieldType::Audio),
            FieldType::Video => Some(AIFieldType::Video),
            FieldType::TimeSeries => Some(AIFieldType::TimeSeries),
            FieldType::Embedding(dim) => Some(AIFieldType::Embedding(*dim)),
            FieldType::Custom(name) => Some(AIFieldType::Custom(name.clone())),
            
            // 这些类型主要用于存储，不直接用于AI处理
            FieldType::Boolean | FieldType::Date | FieldType::Time | FieldType::DateTime | 
            FieldType::Json | FieldType::Object | FieldType::Binary | FieldType::Array(_) => None,
        }
    }
    
    /// 检查是否为AI类型
    pub fn is_ai_type(&self) -> bool {
        matches!(self, 
            FieldType::Numeric | FieldType::Text | FieldType::Categorical |
            FieldType::Image | FieldType::Audio | FieldType::Video |
            FieldType::TimeSeries | FieldType::Embedding(_)
        )
    }
    
    /// 检查是否为存储类型
    pub fn is_storage_type(&self) -> bool {
        matches!(self,
            FieldType::Int | FieldType::Integer | FieldType::Float | FieldType::String |
            FieldType::Boolean | FieldType::Date | FieldType::Time | FieldType::DateTime |
            FieldType::Json | FieldType::Object | FieldType::Binary | FieldType::Array(_)
        )
    }
}

// 字段来源
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldSource {
    /// 来自数据源
    Source,
    /// 来自文件
    File,
    /// 来自数据库
    Database,
    /// 来自计算
    Computed,
    /// 来自用户输入
    UserInput,
    /// 其他来源
    Other(String),
}

// 数据字段定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataField {
    /// 字段名
    pub name: String,
    /// 字段类型
    pub field_type: FieldType,
    /// 是否必填
    pub required: bool,
    /// 字段来源
    pub source: FieldSource,
    /// 字段描述
    pub description: Option<String>,
    /// 默认值
    pub default_value: Option<String>,
}

impl DataField {
    /// 获取nullable属性 (与required相反)
    pub fn nullable(&self) -> bool {
        !self.required
    }
    
    /// 设置nullable属性
    pub fn set_nullable(&mut self, nullable: bool) {
        self.required = !nullable;
    }
}

impl DataField {
    /// 创建新的数据字段
    pub fn new(name: &str, field_type: FieldType, required: bool, source: FieldSource) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required,
            source,
            description: None,
            default_value: None,
        }
    }
    
    /// 创建新的数据字段（使用nullable参数）
    pub fn new_nullable(name: &str, field_type: FieldType, nullable: bool, source: FieldSource) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required: !nullable,
            source,
            description: None,
            default_value: None,
        }
    }
    
    /// 添加描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }
    
    /// 添加默认值
    pub fn with_default(mut self, default_value: &str) -> Self {
        self.default_value = Some(default_value.to_string());
        self
    }
}

// 字段约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConstraints {
    pub required: bool,
    pub unique: bool,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub pattern: Option<String>,
    pub enum_values: Option<Vec<String>>,
    pub default_value: Option<String>,
}

// 字段定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: FieldType,
    pub description: Option<String>,
    pub constraints: Option<FieldConstraints>,
    pub metadata: HashMap<String, String>,
}

// 使用schema.rs中的DataSchema定义
pub use self::schema::DataSchema;

// 索引定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub fields: Vec<String>,
    pub index_type: IndexType,
    pub unique: bool,
}

// 索引类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    FullText,
    Spatial,
    Custom(String),
}

// DataSchema的实现在schema.rs中

// 提供Schema类型别名，用于向下兼容
pub type Schema = DataSchema;

// 提供DataType类型别名，用于向下兼容
pub type DataType = FieldType;

// 提供兼容的Field结构体，用于向下兼容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub data_type: FieldType,
    pub nullable: bool,
    pub metadata: HashMap<String, String>,
}

impl Field {
    pub fn new(name: String, data_type: FieldType, nullable: bool) -> Self {
        Self {
            name,
            data_type,
            nullable,
            metadata: HashMap::new(),
        }
    }
}

// 提供SchemaField类型别名，用于向下兼容
pub type SchemaField = FieldDefinition;

// 从FieldDefinition转换为Field
impl From<FieldDefinition> for Field {
    fn from(field_def: FieldDefinition) -> Self {
        Self {
            name: field_def.name,
            data_type: field_def.field_type,
            nullable: field_def.constraints.as_ref().map(|c| !c.required).unwrap_or(true),
            metadata: field_def.metadata,
        }
    }
}

// 从Field转换为FieldDefinition
impl From<Field> for FieldDefinition {
    fn from(field: Field) -> Self {
        Self {
            name: field.name,
            field_type: field.data_type,
            description: None,
            constraints: Some(FieldConstraints {
                required: !field.nullable,
                unique: false,
                min_value: None,
                max_value: None,
                pattern: None,
                enum_values: None,
                default_value: None,
            }),
            metadata: field.metadata,
        }
    }
}

// 添加SchemaMetadata类型
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SchemaMetadata {
    pub name: String,
    pub description: Option<String>,
    pub version: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub properties: std::collections::HashMap<String, String>,
}

impl Default for SchemaMetadata {
    fn default() -> Self {
        Self {
            name: "未命名模式".to_string(),
            description: None,
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            properties: std::collections::HashMap::new(),
        }
    }
}

// 定义一些模式相关的常量
pub struct SchemaIndex;
pub struct SchemaRelationship;
pub struct RelationshipType;

// Schema推断器接口
pub trait SchemaInferrer: Send + Sync {
    /// 从数据记录推断schema
    fn infer_schema(&self, records: &[crate::data::record::Record]) -> Result<DataSchema>;
    
    /// 获取推断器名称
    fn name(&self) -> &str;
    
    /// 获取推断器描述
    fn description(&self) -> Option<&str> {
        None
    }
}

/// 默认的Schema推断器实现
pub struct DefaultSchemaInferrer {
    name: String,
    sample_size: usize,
    confidence_threshold: f64,
}

impl DefaultSchemaInferrer {
    pub fn new(name: String, sample_size: usize, confidence_threshold: f64) -> Self {
        Self {
            name,
            sample_size,
            confidence_threshold,
        }
    }
}

impl SchemaInferrer for DefaultSchemaInferrer {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> Option<&str> {
        Some("默认的Schema推断器，基于数据样本分析来推断字段类型和约束")
    }
    
    fn infer_schema(&self, records: &[crate::data::record::Record]) -> Result<DataSchema> {
        if records.is_empty() {
            return Err(Error::Validation("无法从空记录集推断schema".to_string()));
        }
        
        // 收集所有字段名
        let mut field_names = std::collections::HashSet::new();
        for record in records {
            for key in record.fields.keys() {
                field_names.insert(key.clone());
            }
        }
        
        let mut fields = Vec::new();
        
        // 分析每个字段
        for field_name in field_names {
            // 收集该字段的所有非空值，转换为 DataValue
            let data_values: Vec<_> = records.iter()
                .filter_map(|r| r.fields.get(&field_name))
                .filter_map(|v| {
                    match v {
                        crate::data::record::Value::Data(dv) => {
                            if !matches!(dv, crate::data::value::DataValue::Null) {
                                Some(dv)
                            } else {
                                None
                            }
                        },
                        _ => None, // 忽略非 Data 类型的值
                    }
                })
                .collect();
            
            // 推断字段类型（infer_field_type 期望 &[&DataValue]）
            let field_type = self.infer_field_type(&data_values.iter().map(|v| *v).collect::<Vec<_>>());
            
            // 创建字段定义（使用 schema.rs 中的 FieldDefinition）
            // 确保使用 schema::FieldType
            let field = schema::FieldDefinition {
                name: field_name,
                field_type: field_type, // infer_field_type 返回 schema::FieldType
                data_type: None,
                required: false,
                nullable: true,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            };
            
            fields.push(field);
        }
        
        // 创建Schema
        let schema = DataSchema {
            name: "inferred_schema".to_string(),
            version: "1.0".to_string(),
            description: None,
            fields,
            primary_key: None, // 不推断主键
            indexes: None,     // 不推断索引
            relationships: None,
            metadata: HashMap::new(),
        };
        
        Ok(schema)
    }
}

impl DefaultSchemaInferrer {
    fn infer_field_type(&self, values: &[&crate::data::value::DataValue]) -> schema::FieldType {
        if values.is_empty() {
            return schema::FieldType::Text; // 默认为文本类型
        }
        
        // 统计每种类型的出现次数
        let mut type_counts = HashMap::new();
        for value in values {
            let type_name = match value {
                crate::data::value::DataValue::Integer(_) => "integer",
                crate::data::value::DataValue::Float(_) => "float",
                crate::data::value::DataValue::String(_) => "string",
                crate::data::value::DataValue::Boolean(_) => "boolean",
                crate::data::value::DataValue::DateTime(_) => "datetime",
                crate::data::value::DataValue::Binary(_) => "binary",
                crate::data::value::DataValue::Array(_) => "array",
                crate::data::value::DataValue::Object(_) => "object",
                crate::data::value::DataValue::Null => continue, // 忽略空值
                _ => "unknown",
            };
            
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
        
        // 找出出现最多的类型
        let (most_common_type, _) = type_counts.iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((&"string", &0));
        
        // 转换为 schema::FieldType
        match *most_common_type {
            "integer" => schema::FieldType::Numeric,
            "float" => schema::FieldType::Numeric,
            "string" => schema::FieldType::Text,
            "boolean" => schema::FieldType::Boolean,
            "datetime" => schema::FieldType::DateTime,
            "binary" => schema::FieldType::Image, // 二进制数据映射为 Image
            "array" => schema::FieldType::Array(Box::new(schema::FieldType::Text)), // 默认为文本数组
            "object" => schema::FieldType::Object(HashMap::new()), // 对象类型
            _ => schema::FieldType::Text, // 默认为文本类型
        }
    }
    
    fn infer_constraints(&self, values: &[&crate::data::value::DataValue]) -> Option<FieldConstraints> {
        if values.is_empty() {
            return None;
        }
        
        let unique = self.is_unique(values);
        
        // 暂时只推断唯一性和必填性
        Some(FieldConstraints {
            required: false, // 默认不必填
            unique,
            min_value: None,
            max_value: None,
            pattern: None,
            enum_values: None,
            default_value: None,
        })
    }
    
    fn is_unique(&self, values: &[&crate::data::value::DataValue]) -> bool {
        let mut unique_values = std::collections::HashSet::new();
        for value in values {
            let serialized = match serde_json::to_string(value) {
                Ok(s) => s,
                Err(_) => return false, // 如果无法序列化，假设不唯一
            };
            
            if !unique_values.insert(serialized) {
                return false; // 找到重复值
            }
        }
        true
    }
}

/// 创建默认的Schema推断器
pub fn create_default_schema_inferrer() -> Result<Box<dyn SchemaInferrer>> {
    Ok(Box::new(DefaultSchemaInferrer::new(
        "default_inferrer".to_string(),
        1000, // 默认样本大小
        0.95, // 默认置信度阈值
    )))
}

/// 创建指定类型的Schema推断器
pub fn create_schema_inferrer(inferrer_type: &str, options: &std::collections::HashMap<String, String>) -> Result<Box<dyn SchemaInferrer>> {
    match inferrer_type {
        "default" => create_default_schema_inferrer(),
        "custom" => {
            // 在这里可以根据options创建自定义的推断器
            Err(Error::NotImplemented("自定义Schema推断器尚未实现".to_string()))
        },
        _ => Err(Error::invalid_argument(format!("不支持的Schema推断器类型: {}", inferrer_type))),
    }
}

/// 检查列是否为数值类型
pub fn is_numeric_column(schema: &DataSchema, column_name: &str) -> bool {
    // 查找列定义
    if let Some(field) = schema.fields.iter().find(|f| f.name == column_name) {
        // 判断字段类型是否为数值类型
        match field.field_type {
            schema::FieldType::Numeric => true,
            _ => false,
        }
    } else {
        false
    }
}


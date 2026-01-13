use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::{Error, Result};
use regex;

/// 字段类型
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum FieldType {
    Numeric,
    Categorical,
    Text,
    Image,
    Audio,
    Video,
    DateTime,
    Boolean,
    Array(Box<FieldType>),
    Object(HashMap<String, FieldType>),
    Custom(String),
}

/// 字段定义
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: FieldType,
    /// 字段数据类型(用于兼容性)
    pub data_type: Option<String>,
    pub required: bool,
    /// 是否可为空
    pub nullable: bool,
    /// 是否为主键
    pub primary_key: bool,
    /// 外键引用
    pub foreign_key: Option<String>,
    pub description: Option<String>,
    pub default_value: Option<String>,
    pub constraints: Option<FieldConstraints>,
    pub metadata: HashMap<String, String>,
}

impl FieldDefinition {
    /// 获取字段名
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 获取字段类型
    pub fn field_type(&self) -> &FieldType {
        &self.field_type
    }

    /// 获取兼容性数据类型
    pub fn data_type(&self) -> Option<&String> {
        self.data_type.as_ref()
    }

    /// 获取值类型（与 field_type 等价，兼容旧接口）
    pub fn value_type(&self) -> &FieldType {
        &self.field_type
    }

    /// 设置字段类型
    pub fn set_field_type(&mut self, field_type: FieldType) {
        self.field_type = field_type;
    }
}

/// 字段约束
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub allowed_values: Option<Vec<String>>,
    pub unique: bool,
}

/// 数据模式
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSchema {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub fields: Vec<FieldDefinition>,
    pub primary_key: Option<Vec<String>>,
    pub indexes: Option<Vec<SchemaIndex>>,
    pub relationships: Option<Vec<SchemaRelationship>>,
    /// 模式级元数据
    pub metadata: HashMap<String, String>,
}

/// 模式索引
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaIndex {
    pub name: String,
    pub fields: Vec<String>,
    pub unique: bool,
}

/// 模式关系
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaRelationship {
    pub name: String,
    pub from_field: String,
    pub to_schema: String,
    pub to_field: String,
    pub relationship_type: RelationshipType,
}

/// 关系类型
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

/// 数据模式构建器
pub struct DataSchemaBuilder {
    name: Option<String>,
    version: Option<String>,
    description: Option<String>,
    fields: Vec<FieldDefinition>,
    primary_key: Option<Vec<String>>,
    indexes: Option<Vec<SchemaIndex>>,
    relationships: Option<Vec<SchemaRelationship>>,
}

impl DataSchemaBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            name: None,
            version: None,
            description: None,
            fields: Vec::new(),
            primary_key: None,
            indexes: None,
            relationships: None,
        }
    }

    /// 设置名称
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// 设置版本
    pub fn version(mut self, version: &str) -> Self {
        self.version = Some(version.to_string());
        self
    }

    /// 设置描述
    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// 添加字段
    pub fn field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// 构建DataSchema
    pub fn build(self) -> Result<DataSchema> {
        let name = self.name.unwrap_or_else(|| "unnamed_schema".to_string());
        let version = self.version.unwrap_or_else(|| "1.0.0".to_string());
        
        Ok(DataSchema {
            name,
            version,
            description: self.description,
            fields: self.fields,
            primary_key: self.primary_key,
            indexes: self.indexes,
            relationships: self.relationships,
            metadata: HashMap::new(),
        })
    }
}

impl Default for DataSchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSchema {
    /// 创建新的数据模式
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            description: None,
            fields: Vec::new(),
            primary_key: None,
            indexes: None,
            relationships: None,
            metadata: HashMap::new(),
        }
    }

    /// 获取字段列表
    pub fn fields(&self) -> &Vec<FieldDefinition> {
        &self.fields
    }

    /// 根据字段名获取字段定义
    pub fn get_field(&self, name: &str) -> Option<&FieldDefinition> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// 判断是否包含某字段
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    /// 兼容 contains_key 调用
    pub fn contains_key(&self, name: &str) -> bool {
        self.has_field(name)
    }

    /// 创建模式构建器
    pub fn builder() -> DataSchemaBuilder {
        DataSchemaBuilder::new()
    }
    
    /// 添加描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }
    
    /// 添加字段
    pub fn add_field(&mut self, field: FieldDefinition) -> Result<()> {
        // 检查字段名是否已存在
        if self.fields.iter().any(|f| f.name == field.name) {
            return Err(Error::InvalidArgument(format!(
                "Field {} already exists in schema", field.name
            )));
        }
        
        self.fields.push(field);
        Ok(())
    }
    
    /// 添加多个字段
    pub fn add_fields(&mut self, fields: Vec<FieldDefinition>) -> Result<()> {
        for field in fields {
            self.add_field(field)?;
        }
        Ok(())
    }

    /// 设置主键
    pub fn set_primary_key(&mut self, fields: Vec<String>) -> Result<()> {
        // 验证字段是否存在
        for field in &fields {
            if !self.fields.iter().any(|f| f.name == *field) {
                return Err(Error::InvalidArgument(format!(
                    "Primary key field {} does not exist in schema", field
                )));
            }
        }
        
        self.primary_key = Some(fields);
        Ok(())
    }
    
    /// 添加索引
    pub fn add_index(&mut self, index: SchemaIndex) -> Result<()> {
        // 验证字段是否存在
        for field in &index.fields {
            if !self.fields.iter().any(|f| f.name == *field) {
                return Err(Error::InvalidArgument(format!(
                    "Index field {} does not exist in schema", field
                )));
            }
        }
        
        if self.indexes.is_none() {
            self.indexes = Some(Vec::new());
        }
        
        self.indexes.as_mut().unwrap().push(index);
        Ok(())
    }
    
    /// 添加关系
    pub fn add_relationship(&mut self, relationship: SchemaRelationship) -> Result<()> {
        // 验证字段是否存在
        if !self.fields.iter().any(|f| f.name == relationship.from_field) {
            return Err(Error::InvalidArgument(format!(
                "Relationship field {} does not exist in schema", relationship.from_field
            )));
        }
        
        if self.relationships.is_none() {
            self.relationships = Some(Vec::new());
        }
        
        self.relationships.as_mut().unwrap().push(relationship);
        Ok(())
    }
    
    /// 验证数据
    pub fn validate(&self, data: &HashMap<String, serde_json::Value>) -> Result<()> {
        // 验证必填字段
        for field in &self.fields {
            if field.required && !data.contains_key(&field.name) {
                return Err(Error::Validation(format!(
                    "Required field {} is missing", field.name
                )));
            }
        }
        
        // 验证字段类型和约束
        for (name, value) in data {
            if let Some(field) = self.fields.iter().find(|f| f.name == *name) {
                self.validate_field_value(field, value)?;
            } else {
                // 未知字段，忽略
            }
        }
        
        Ok(())
    }
    
    /// 验证字段值
    fn validate_field_value(&self, field: &FieldDefinition, value: &serde_json::Value) -> Result<()> {
        match &field.field_type {
            FieldType::Numeric => {
                if !value.is_number() {
                    return Err(Error::Validation(format!(
                        "Field {} must be a number", field.name
                    )));
                }
                
                // 验证约束
                if let Some(constraints) = &field.constraints {
                    let num_value = value.as_f64().unwrap();
                    
                    if let Some(min_value) = constraints.min_value {
                        if num_value < min_value {
                            return Err(Error::Validation(format!(
                                "Field {} value {} is less than minimum {}", field.name, num_value, min_value
                            )));
                        }
                    }
                    
                    if let Some(max_value) = constraints.max_value {
                        if num_value > max_value {
                            return Err(Error::Validation(format!(
                                "Field {} value {} is greater than maximum {}", field.name, num_value, max_value
                            )));
                        }
                    }
                }
            },
            FieldType::Categorical | FieldType::Text => {
                if !value.is_string() {
                    return Err(Error::Validation(format!(
                        "Field {} must be a string", field.name
                    )));
                }
                
                let str_value = value.as_str().unwrap();
                
                // 验证约束
                if let Some(constraints) = &field.constraints {
                    if let Some(min_length) = constraints.min_length {
                        if str_value.len() < min_length {
                            return Err(Error::Validation(format!(
                                "Field {} length {} is less than minimum {}", field.name, str_value.len(), min_length
                            )));
                        }
                    }
                    
                    if let Some(max_length) = constraints.max_length {
                        if str_value.len() > max_length {
                            return Err(Error::Validation(format!(
                                "Field {} length {} is greater than maximum {}", field.name, str_value.len(), max_length
                            )));
                        }
                    }
                    
                    if let Some(pattern) = &constraints.pattern {
                        let re = regex::Regex::new(pattern)?;
                        if !re.is_match(str_value) {
                            return Err(Error::Validation(format!(
                                "Field {} value '{}' does not match pattern '{}'", field.name, str_value, pattern
                            )));
                        }
                    }
                    
                    if let Some(allowed_values) = &constraints.allowed_values {
                        if !allowed_values.contains(&str_value.to_string()) {
                            return Err(Error::Validation(format!(
                                "Field {} value '{}' is not in allowed values: {:?}", field.name, str_value, allowed_values
                            )));
                        }
                    }
                }
            },
            FieldType::Boolean => {
                if !value.is_boolean() {
                    return Err(Error::Validation(format!(
                        "Field {} must be a boolean", field.name
                    )));
                }
            },
            FieldType::DateTime => {
                if !value.is_string() {
                    return Err(Error::Validation(format!(
                        "Field {} must be a string representing a date time", field.name
                    )));
                }
                
                // 注意：这里需要使用日期时间库验证格式
                // 为简化示例，我们跳过日期时间验证
            },
            FieldType::Array(item_type) => {
                if !value.is_array() {
                    return Err(Error::Validation(format!(
                        "Field {} must be an array", field.name
                    )));
                }
                
                // 验证数组元素
                let array = value.as_array().unwrap();
                
                // 验证约束
                if let Some(constraints) = &field.constraints {
                    if let Some(min_length) = constraints.min_length {
                        if array.len() < min_length {
                            return Err(Error::Validation(format!(
                                "Field {} length {} is less than minimum {}", field.name, array.len(), min_length
                            )));
                        }
                    }
                    
                    if let Some(max_length) = constraints.max_length {
                        if array.len() > max_length {
                            return Err(Error::Validation(format!(
                                "Field {} length {} is greater than maximum {}", field.name, array.len(), max_length
                            )));
                        }
                    }
                }
                
                // 验证数组元素类型
                // 为简化示例，我们跳过数组元素类型验证
            },
            FieldType::Object(properties) => {
                if !value.is_object() {
                    return Err(Error::Validation(format!(
                        "Field {} must be an object", field.name
                    )));
                }
                
                // 验证对象属性
                // 为简化示例，我们跳过对象属性验证
            },
            _ => {
                // 其他类型暂不支持验证
                return Err(Error::Validation(format!(
                    "Unsupported field type: {:?}", field.field_type
                )));
            }
        }
        
        Ok(())
    }
    
    /// 从JSON加载模式
    pub fn from_json(json: &str) -> Result<Self> {
        let schema: DataSchema = serde_json::from_str(json)?;
        Ok(schema)
    }
    
    /// 转换为JSON
    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }
    
    /// 创建带字段的数据模式
    pub fn new_with_fields(fields: Vec<FieldDefinition>, name: &str, version: &str) -> Self {
        let mut schema = Self {
            name: name.to_string(),
            version: version.to_string(),
            description: None,
            fields: Vec::new(),
            primary_key: None,
            indexes: None,
            relationships: None,
            metadata: HashMap::new(),
        };
        
        // 添加字段，忽略可能的错误
        let _ = schema.add_fields(fields);
        
        schema
    }

    /// 获取字段名称列表
    pub fn field_names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// 获取字段类型列表
    pub fn field_types(&self) -> Vec<FieldType> {
        self.fields.iter().map(|f| f.field_type.clone()).collect()
    }

    /// 获取特征字段列表（当前等同于全部字段）
    pub fn feature_fields(&self) -> Vec<String> {
        self.field_names()
    }

    /// 获取元数据字段列表（当前返回空）
    pub fn metadata_fields(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }
}

impl Default for DataSchema {
    fn default() -> Self {
        DataSchema::new("default", "1.0")
    }
}

/// 创建字段定义
pub fn field(name: &str, field_type: FieldType, required: bool) -> FieldDefinition {
    FieldDefinition {
        name: name.to_string(),
        field_type,
        data_type: None,
        required,
        nullable: false,
        primary_key: false,
        foreign_key: None,
        description: None,
        default_value: None,
        constraints: None,
        metadata: HashMap::new(),
    }
}

/// 创建字段约束
pub fn constraints() -> FieldConstraints {
    FieldConstraints {
        min_value: None,
        max_value: None,
        min_length: None,
        max_length: None,
        pattern: None,
        allowed_values: None,
        unique: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_schema_creation() {
        // 创建模式
        let mut schema = DataSchema::new("user", "1.0")
            .with_description("User schema");
        
        // 添加字段
        schema.add_field(field("id", FieldType::Text, true)).unwrap();
        schema.add_field(field("name", FieldType::Text, true)).unwrap();
        schema.add_field(field("age", FieldType::Numeric, false)).unwrap();
        
        // 设置主键
        schema.set_primary_key(vec!["id".to_string()]).unwrap();
        
        // 添加索引
        schema.add_index(SchemaIndex {
            name: "name_index".to_string(),
            fields: vec!["name".to_string()],
            unique: false,
        }).unwrap();
        
        // 验证模式
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.primary_key, Some(vec!["id".to_string()]));
        assert_eq!(schema.indexes.as_ref().unwrap().len(), 1);
    }
    
    #[test]
    fn test_schema_validation() {
        // 创建模式
        let mut schema = DataSchema::new("user", "1.0");
        
        // 添加字段
        schema.add_field(field("id", FieldType::Text, true)).unwrap();
        schema.add_field(field("name", FieldType::Text, true)).unwrap();
        schema.add_field(field("age", FieldType::Numeric, false)).unwrap();
        
        // 创建有效数据
        let mut valid_data = HashMap::new();
        valid_data.insert("id".to_string(), serde_json::json!("123"));
        valid_data.insert("name".to_string(), serde_json::json!("John"));
        valid_data.insert("age".to_string(), serde_json::json!(30));
        
        // 验证有效数据
        assert!(schema.validate(&valid_data).is_ok());
        
        // 创建无效数据（缺少必填字段）
        let mut invalid_data = HashMap::new();
        invalid_data.insert("id".to_string(), serde_json::json!("123"));
        
        // 验证无效数据
        assert!(schema.validate(&invalid_data).is_err());
    }
} 
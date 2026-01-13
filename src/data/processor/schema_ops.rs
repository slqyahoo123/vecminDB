// Data Schema Operations
// 数据模式操作模块

use serde_json::Value;
use std::collections::HashMap;
use crate::Result;
use crate::Error;
use crate::data::processor::types::{Schema, DataType, SimpleSchema};

/// 从元数据中提取数据模式
pub fn extract_schema_from_metadata(metadata: &Value) -> Result<Schema> {
    // 解析元数据中的模式信息
    if let Some(schema_obj) = metadata.get("schema") {
        let mut fields: HashMap<String, DataType> = HashMap::new();

        if let Some(fields_obj) = schema_obj.get("fields") {
            if let Some(fields_array) = fields_obj.as_array() {
                for field in fields_array {
                    if let (Some(name), Some(type_str)) = (field.get("name").and_then(Value::as_str), field.get("type").and_then(Value::as_str)) {
                        let field_type = match type_str {
                            "string" => DataType::String,
                            "integer" | "int" => DataType::Integer,
                            "float" | "number" => DataType::Float,
                            "boolean" | "bool" => DataType::Boolean,
                            "date" | "datetime" => DataType::DateTime,
                            "array" => {
                                let item_type = if let Some(items) = field.get("items") {
                                    if let Some(item_type) = items.get("type").and_then(Value::as_str) {
                                        match item_type {
                                            "string" => DataType::Array(Box::new(DataType::String)),
                                            "integer" | "int" => DataType::Array(Box::new(DataType::Integer)),
                                            "float" | "number" => DataType::Array(Box::new(DataType::Float)),
                                            "boolean" | "bool" => DataType::Array(Box::new(DataType::Boolean)),
                                            _ => DataType::Array(Box::new(DataType::Any)),
                                        }
                                    } else {
                                        DataType::Array(Box::new(DataType::Any))
                                    }
                                } else {
                                    DataType::Array(Box::new(DataType::Any))
                                };
                                item_type
                            },
                            "object" => DataType::Object,
                            _ => DataType::Any,
                        };
                        
                        // 解析数据模式
                        let description = field.get("description").and_then(Value::as_str).map(String::from);
                        let required = field.get("required").and_then(Value::as_bool).unwrap_or(false);
                        
                        // 目前 SimpleSchema 仅存储类型信息，字段级元数据由上层 ProcessorSchema 处理
                        // 这里只记录字段类型
                        let _ = (description, required); // 保留解析逻辑以便后续扩展
                        fields.insert(name.to_string(), field_type);
                    }
                }
            }
        }

        // 创建Schema对象（仅基于字段类型映射）
        let simple = SimpleSchema::from_map(fields, None);
        let schema = Schema::Simple(simple);
        return Ok(schema);
    }
    
    // 如果无法解析元数据，返回错误
    Err(Error::invalid_data("无法解析元数据中的模式信息"))
}

/// 从数据批次中推断数据模式
pub fn infer_schema_from_data(data: &[Value]) -> Result<Schema> {
    if data.is_empty() {
        return Err(Error::invalid_data("无法从空数据中推断模式"));
    }
    
    // 遍历数据，推断模式（仅记录字段类型）
    let mut fields: HashMap<String, DataType> = HashMap::new();
    
    for (i, item) in data.iter().enumerate() {
        if let Some(obj) = item.as_object() {
            for (key, value) in obj {
                // 如果字段已存在，则推断兼容类型
                if let Some(existing_type) = fields.get_mut(key) {
                    let inferred_type = infer_type_from_value(value);
                    
                    // 更新兼容类型
                    *existing_type = get_compatible_type(existing_type, &inferred_type);
                } else {
                    // 如果字段不存在，则推断类型并插入
                    let inferred_type = infer_type_from_value(value);
                    fields.insert(key.clone(), inferred_type);
                }
            }
        }
    }
    
    Ok(Schema::Simple(SimpleSchema::from_map(fields, None)))
}

/// 从值推断类型
fn infer_type_from_value(value: &Value) -> DataType {
    match value {
        Value::Null => DataType::Null,
        Value::Bool(_) => DataType::Boolean,
        Value::Number(n) => {
            if n.is_i64() {
                DataType::Integer
            } else {
                DataType::Float
            }
        },
        Value::String(s) => {
            // 尝试解析为日期时间
            if s.contains("T") && (s.contains("Z") || s.contains("+")) && s.len() >= 20 {
                DataType::DateTime
            } else {
                DataType::String
            }
        },
        Value::Array(arr) => {
            if arr.is_empty() {
                DataType::Array(Box::new(DataType::Any))
            } else {
                // 推断数组元素类型
                let element_type = infer_type_from_value(&arr[0]);
                DataType::Array(Box::new(element_type))
            }
        },
        Value::Object(_) => DataType::Object,
    }
}

/// 获取两种类型的兼容类型
fn get_compatible_type(type1: &DataType, type2: &DataType) -> DataType {
    match (type1, type2) {
        // 相同类型保持不变
        (t1, t2) if t1 == t2 => t1.clone(),
        
        // 空值与任何类型兼容
        (DataType::Null, t) => t.clone(),
        (t, DataType::Null) => t.clone(),
        
        // 整数和浮点数提升为浮点数
        (DataType::Integer, DataType::Float) | 
        (DataType::Float, DataType::Integer) => DataType::Float,
        
        // 数组类型
        (DataType::Array(t1), DataType::Array(t2)) => {
            let compatible = get_compatible_type(t1, t2);
            DataType::Array(Box::new(compatible))
        },
        
        // 其他情况退化为Any类型
        _ => DataType::Any,
    }
}

/// 将Schema转换为SimpleSchema
pub fn convert_to_simple_schema(schema: &Schema) -> SimpleSchema {
    // 直接使用 SchemaAdapter 暴露的字段类型映射
    let simple_fields = schema.fields();
    SimpleSchema::from_map(simple_fields, None)
}

/// 验证数据是否符合模式
pub fn validate_data_against_schema(data: &Value, schema: &Schema) -> Result<()> {
    if let Some(obj) = data.as_object() {
        for (field_name, expected_type) in schema.fields() {
            // 当前 SchemaAdapter 不承载 required 等元数据，这里仅验证类型兼容性
            if let Some(value) = obj.get(&field_name) {
                if !is_value_compatible_with_type(value, &expected_type) {
                    return Err(Error::validation(&format!(
                        "字段 {} 的类型不匹配: 期望 {:?}, 实际值 {:?}",
                        field_name, expected_type, value
                    )));
                }
            }
        }
        Ok(())
    } else {
        Err(Error::validation("数据必须是一个对象"))
    }
}

/// 检查值类型兼容性
fn is_value_compatible_with_type(value: &Value, expected_type: &DataType) -> bool {
    match (value, expected_type) {
        (Value::Null, DataType::Null) => true,
        (Value::Null, _) => true, // 空值与任何类型兼容
        
        (Value::Bool(_), DataType::Boolean) => true,
        
        (Value::Number(n), DataType::Integer) => n.is_i64(),
        (Value::Number(_), DataType::Float) => true,
        (Value::Number(_), DataType::Any) => true,
        
        (Value::String(_), DataType::String) => true,
        (Value::String(_), DataType::DateTime) => true, // 应该有更严格的验证
        (Value::String(_), DataType::Any) => true,
        
        (Value::Array(arr), DataType::Array(element_type)) => {
            arr.iter().all(|item| is_value_compatible_with_type(item, element_type))
        },
        (Value::Array(_), DataType::Any) => true,
        
        (Value::Object(_), DataType::Object) => true,
        (Value::Object(_), DataType::Any) => true,
        
        (_, DataType::Any) => true,
        
        _ => false,
    }
}

/// 模式合并策略
pub enum MergeStrategy {
    /// 使用最严格的类型
    Strict,
    /// 使用最宽松的类型
    Relaxed,
    /// 使用最常见的类型
    MostCommon,
}

/// 创建数据模式合并工具
pub struct SchemaMerger {
    schemas: Vec<Schema>,
    strategy: MergeStrategy,
}

impl SchemaMerger {
    /// 创建新的模式合并工具
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            schemas: Vec::new(),
            strategy,
        }
    }
    
    /// 添加模式
    pub fn add_schema(&mut self, schema: Schema) {
        self.schemas.push(schema);
    }
    
    /// 合并所有模式
    pub fn merge(&self) -> Result<Schema> {
        if self.schemas.is_empty() {
            return Err(Error::invalid_data("没有模式可合并"));
        }
        
        // 收集所有字段
        let mut merged_fields: HashMap<String, Vec<DataType>> = HashMap::new();
        
        // 收集所有模式中的所有字段
        for schema in &self.schemas {
            for (field_name, field_type) in schema.fields() {
                merged_fields
                    .entry(field_name.clone())
                    .or_insert_with(Vec::new)
                    .push(field_type.clone());
            }
        }
        
        // 根据策略合并字段
        let mut final_fields: HashMap<String, DataType> = HashMap::new();
        
        for (field_name, field_infos) in merged_fields {
            let merged_type = match self.strategy {
                MergeStrategy::Strict => self.get_strictest_type(&field_infos),
                MergeStrategy::Relaxed => self.get_most_relaxed_type(&field_infos),
                MergeStrategy::MostCommon => self.get_most_common_type(&field_infos),
            };
            
            // 当前仅按类型合并，不合并元数据
            final_fields.insert(field_name, merged_type);
        }
        
        Ok(Schema::Simple(SimpleSchema::from_map(final_fields, None)))
    }
    
    /// 获取最严格的类型
    fn get_strictest_type(&self, field_infos: &[DataType]) -> DataType {
        if field_infos.is_empty() {
            return DataType::Any;
        }
        
        let mut result = field_infos[0].clone();
        
        for field_type in field_infos.iter().skip(1) {
            result = match (&result, field_type) {
                // 保留更具体的类型
                (DataType::Any, t) => t.clone(),
                (t, DataType::Any) => t.clone(),
                
                // 数字类型处理
                (DataType::Float, DataType::Integer) => DataType::Integer,
                (DataType::Integer, DataType::Float) => DataType::Integer,
                
                // 数组类型
                (DataType::Array(t1), DataType::Array(t2)) => {
                    let strict_inner = self.get_strictest_type(&[(*t1.as_ref()).clone(), (*t2.as_ref()).clone()]);
                    DataType::Array(Box::new(strict_inner))
                },
                
                // 类型不匹配时，选择更具体的类型
                (t1, t2) if t1 == t2 => t1.clone(),
                
                // 默认保持第一个类型
                _ => result.clone(),
            };
        }
        
        result
    }
    
    /// 获取最宽松的类型
    fn get_most_relaxed_type(&self, field_infos: &[DataType]) -> DataType {
        if field_infos.is_empty() {
            return DataType::Any;
        }
        
        let mut result = field_infos[0].clone();
        
        for field_type in field_infos.iter().skip(1) {
            result = match (&result, field_type) {
                // Any 是最宽松的类型
                (DataType::Any, _) | (_, DataType::Any) => DataType::Any,
                
                // 数字类型处理
                (DataType::Integer, DataType::Float) | 
                (DataType::Float, DataType::Integer) => DataType::Float,
                
                // 数组类型
                (DataType::Array(t1), DataType::Array(t2)) => {
                    let relaxed_inner = self.get_most_relaxed_type(&[(*t1.as_ref()).clone(), (*t2.as_ref()).clone()]);
                    DataType::Array(Box::new(relaxed_inner))
                },
                
                // 类型完全匹配
                (t1, t2) if t1 == t2 => t1.clone(),
                
                // 不匹配时选择Any
                _ => DataType::Any,
            };
        }
        
        result
    }
    
    /// 获取最常见的类型
    fn get_most_common_type(&self, field_infos: &[DataType]) -> DataType {
        if field_infos.is_empty() {
            return DataType::Any;
        }
        
        // 统计类型频率
        let mut type_counts = HashMap::new();
        for field_type in field_infos {
            *type_counts.entry(field_type).or_insert(0) += 1;
        }
        
        // 找出最常见的类型
        let mut max_count = 0;
        let mut most_common_type = &DataType::Any;
        
        for (data_type, count) in type_counts.iter() {
            if *count > max_count {
                max_count = *count;
                most_common_type = *data_type;
            }
        }
        
        most_common_type.clone()
    }
    
}

/// 比较两个模式的差异
pub fn compare_schemas(schema1: &Schema, schema2: &Schema) -> Vec<String> {
    let mut differences = Vec::new();
    
    // 检查schema1中存在但schema2中不存在的字段
    for (field_name, field_type1) in schema1.fields() {
        if let Some(field_type2) = schema2.fields().get(&field_name) {
            if field_type1 != *field_type2 {
                differences.push(format!(
                    "字段 {} 类型不同: {:?} vs {:?}",
                    field_name, field_type1, field_type2
                ));
            }
        } else {
            differences.push(format!("字段 {} 在第二个模式中不存在", field_name));
        }
    }
    
    // 检查schema2中存在但schema1中不存在的字段
    for field_name in schema2.fields().keys() {
        if !schema1.fields().contains_key(field_name) {
            differences.push(format!("字段 {} 在第一个模式中不存在", field_name));
        }
    }
    
    differences
}

/// 生成模式统计信息
pub fn generate_schema_statistics(schema: &Schema) -> HashMap<String, Value> {
    let mut stats = HashMap::new();
    
    stats.insert("total_fields".to_string(), Value::from(schema.fields().len()));
    
    // 统计各种类型的字段数量
    let mut type_counts = HashMap::new();
    
    for (_, field_type) in schema.fields() {
        let type_name = format!("{:?}", field_type);
        *type_counts.entry(type_name).or_insert(0) += 1;
    }
    
    stats.insert("type_distribution".to_string(), serde_json::to_value(type_counts).unwrap_or_default());
    
    stats
} 